import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer

from src.teacher.ilora_model import iLoRAModel
from src.teacher.trainer_ilora import iLoRATrainer
from src.student.datamodule import SASRecDataModule
from src.teacher.mlp_projector import MLPProjector

@pytest.fixture(scope="module")
def ilora_env():
    """
    Sets up the environment for testing iLoRA optimizations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    ilora_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 6, # Adjusted for divisibility by num_lora_experts (6 % 3 == 0)
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })

    # Load LLM and Tokenizer
    # Optimization 1: Use AutoModel instead of AutoModelForCausalLM
    llm = AutoModel.from_pretrained(ilora_cfg.llm_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ilora_cfg.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))

    # Mock DataModule
    class MockSASRecDataModule(pl.LightningDataModule):
        def __init__(self, tokenizer, batch_size=2, max_seq_len=20):
            super().__init__()
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_seq_len = max_seq_len
            self.num_items = 100
            self.padding_item_id = 0
            self.mapped_id_to_title = {i: str(i) for i in range(self.num_items + 1)}

        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            # Create dummy batch
            input_ids = torch.randint(0, len(self.tokenizer), (self.batch_size, self.max_seq_len))
            attention_mask = torch.ones((self.batch_size, self.max_seq_len), dtype=torch.long)
            seq = torch.randint(1, self.num_items, (self.batch_size, self.max_seq_len))
            len_seq = torch.randint(1, self.max_seq_len + 1, (self.batch_size,))
            cans = torch.randint(1, self.num_items, (self.batch_size, 20))
            len_cans = torch.randint(1, 21, (self.batch_size,))
            item_id = torch.randint(1, self.num_items, (self.batch_size,))
            next_item = torch.randint(1, self.num_items, (self.batch_size,))
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "seq": seq,
                "len_seq": len_seq,
                "cans": cans,
                "len_cans": len_cans,
                "item_id": item_id,
                "next_item": next_item
            }
            return torch.utils.data.DataLoader([batch], batch_size=None)

    dm = MockSASRecDataModule(tokenizer=tokenizer)
    dm.setup()

    # Dummy RecModel
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size_rec, num_items_rec, padding_item_id):
            super().__init__()
            self.hidden_size = hidden_size_rec
            self.item_embeddings = nn.Embedding(num_items_rec + 1, hidden_size_rec, padding_idx=padding_item_id)
            self.cacu_x = lambda x: self.item_embeddings(x)
            self.cacul_h = lambda x, y: torch.randn(x.shape[0], hidden_size_rec).to(x.device)
        
        def get_full_sequence_representations(self, item_seq, item_seq_len):
            batch_size, seq_len = item_seq.shape
            return torch.randn(batch_size, seq_len, self.hidden_size).to(item_seq.device)

        def _get_last_item_representation(self, item_seq, item_seq_len):
            batch_size = item_seq.shape[0]
            return torch.randn(batch_size, self.hidden_size).to(item_seq.device)

    dummy_rec_model = DummyRecModel(ilora_cfg.hidden_size, dm.num_items, dm.padding_item_id).to(device)
    dummy_projector = MLPProjector(
        input_dim=ilora_cfg.hidden_size,
        output_dim=llm.config.hidden_size,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate
    ).to(device)

    # Initialize iLoRAModel
    ilora_model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=ilora_cfg.num_lora_experts,
        lora_r=ilora_cfg.lora_r,
        lora_alpha=ilora_cfg.lora_alpha,
        lora_dropout=ilora_cfg.lora_dropout,
        num_items=dm.num_items,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate,
        rec_model=dummy_rec_model,
        projector=dummy_projector,
        candidate_topk=10,
        item_id_to_name=dm.mapped_id_to_title,
        padding_item_id=dm.padding_item_id,
        llm_dtype=torch.float32
    ).to(device)

    # Initialize iLoRATrainer
    ilora_trainer = iLoRATrainer(
        ilora_model=ilora_model,
        num_items=dm.num_items,
        learning_rate=1e-4,
        weight_decay=0.01,
        metrics_k=10,
        item_id_to_name=dm.mapped_id_to_title
    ).to(device)

    return {
        "llm": llm,
        "ilora_model": ilora_model,
        "ilora_trainer": ilora_trainer,
        "datamodule": dm,
        "tokenizer": tokenizer
    }

def test_lm_head_removal(ilora_env):
    """
    Optimization 1: Verify that the model does not have an LM Head.
    """
    llm = ilora_env["llm"]
    ilora_model = ilora_env["ilora_model"]
    
    # Check if the base model is AutoModel (not AutoModelForCausalLM)
    # Note: transformers models are usually specific classes, e.g., OPTModel
    # We check that it does NOT have 'lm_head'
    
    assert not hasattr(llm, "lm_head"), "Base LLM should not have 'lm_head' attribute"
    
    # Also check the wrapped model in iLoRAModel
    # iLoRAModel.llm is MoeLoraModel, which wraps the base model
    assert not hasattr(ilora_model.llm.model, "lm_head"), "Wrapped LLM should not have 'lm_head' attribute"

def test_sampled_softmax_execution(ilora_env):
    """
    Optimization 2: Verify that Sampled Softmax training step runs successfully.
    """
    trainer_model = ilora_env["ilora_trainer"]
    dm = ilora_env["datamodule"]

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1
    )

    # Run training step
    trainer.fit(trainer_model, dm.train_dataloader())
    
    # Check if loss is calculated
    loss = trainer_model.trainer.callback_metrics.get("train_loss_step")
    assert loss is not None, "Training loss should be recorded"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"

def test_prompt_template_update(ilora_env):
    """
    Optimization 3: Verify that the prompt template does not contain [CansEmb].
    """
    # This test would ideally check the DataModule's collate_fn output.
    # However, since we are using a MockSASRecDataModule in this test file,
    # we can't directly verify the real SASRecDataModule's logic here unless we import it.
    # To properly test the optimization, we should instantiate the REAL SASRecDataModule's collater.
    
    from src.student.datamodule import TeacherTrainCollater
    
    tokenizer = ilora_env["tokenizer"]
    collater = TeacherTrainCollater(
        tokenizer=tokenizer,
        max_seq_len=20,
        padding_item_id=0,
        id_to_name={i: str(i) for i in range(100)}
    )
    
    # The prompt template is defined in the collater or used by it.
    # Let's inspect the prompt construction logic if possible, or run a dummy batch through it.
    # Since TeacherTrainCollater.prompt_template is not directly exposed as a class attribute in the snippet I saw,
    # I will check the output of the collater.
    
    # Create dummy instances
    instances = [{
        "seq_ids": [1, 2],
        "next_item_id": 11,
        "history_str": "Movie1, Movie2",
        "candidates_str": "Movie4, Movie5", # Should be ignored by template
        "candidates": [4, 5]
    }]
    
    batch = collater(instances)
    input_ids = batch["input_ids"]
    decoded_text = tokenizer.batch_decode(input_ids)[0]
    
    assert "[CansEmb]" not in decoded_text, "Prompt should not contain [CansEmb]"
    assert "[HistoryEmb]" in decoded_text, "Prompt should contain [HistoryEmb]"
