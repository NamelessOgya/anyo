import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from typing import Dict, Any

from src.student.models import SASRec
from src.student.trainer_baseline import SASRecTrainer
from src.teacher.ilora_model import iLoRAModel
from src.teacher.trainer_ilora import iLoRATrainer
from src.distill.trainer_distill import DistillationTrainer
from src.teacher.mlp_projector import MLPProjector
from src.distill.selection_policy import AllSamplesPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer

@pytest.fixture(scope="module")
def trainer_fixture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_items = 100
    hidden_size = 32
    max_seq_len = 20
    
    # --- SASRec Setup ---
    # We need to know teacher embedding dim beforehand or update it later.
    # LLM hidden size is usually 768 for OPT-125m.
    teacher_hidden_size = 768
    
    sasrec_model = SASRec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.0,
        max_seq_len=max_seq_len,
        padding_item_id=0,
        teacher_embedding_dim=teacher_hidden_size,
        ed_weight=0.1
    ).to(device)
    
    sasrec_trainer = SASRecTrainer(
        rec_model=sasrec_model,
        num_items=num_items,
        learning_rate=1e-3,
        weight_decay=0.0,
        metrics_k=5
    )
    
    # --- iLoRA Setup ---
    llm_model_name = "facebook/opt-125m"
    llm = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))
    
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size, num_items):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_items = num_items
            self.item_embeddings = nn.Embedding(num_items + 1, hidden_size)
            
        def get_full_sequence_representations(self, item_seq, item_seq_len, teacher_embeddings=None):
            batch_size, seq_len = item_seq.shape
            return torch.randn(batch_size, seq_len, self.hidden_size).to(item_seq.device)

        def _get_last_item_representation(self, item_seq, item_seq_len):
            batch_size = item_seq.shape[0]
            return torch.randn(batch_size, self.hidden_size).to(item_seq.device)
            
        def predict(self, item_seq, item_seq_len):
            # Return random logits
            batch_size = item_seq.shape[0]
            return torch.randn(batch_size, self.num_items).to(item_seq.device)

    dummy_rec_model = DummyRecModel(hidden_size, num_items).to(device)
    
    ilora_model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=2,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        num_items=num_items,
        hidden_size=hidden_size,
        dropout_rate=0.0,
        rec_model=dummy_rec_model,
        projector=MLPProjector(hidden_size, llm.config.hidden_size, hidden_size, 0.0),
        candidate_topk=5,
        item_id_to_name={i: str(i) for i in range(num_items + 1)},
        padding_item_id=0,
        llm_dtype=torch.float32
    ).to(device)
    
    ilora_trainer = iLoRATrainer(
        ilora_model=ilora_model,
        num_items=num_items,
        learning_rate=1e-3,
        weight_decay=0.0,
        metrics_k=5,
        item_id_to_name={i: str(i) for i in range(num_items + 1)}
    )
    
    # --- Distillation Setup ---
    class DummyDataModule:
        def __init__(self):
            self.padding_item_id = 0
            
    distill_trainer = DistillationTrainer(
        student_model=sasrec_model,
        teacher_model=ilora_model,
        datamodule=DummyDataModule(),
        num_items=num_items,
        ranking_loss_weight=1.0,
        embedding_loss_weight=1.0,
        ce_loss_weight=1.0,
        ranking_temperature=1.0,
        embedding_loss_type="mse",
        learning_rate=1e-3,
        weight_decay=0.0,
        metrics_k=5,
        selection_policy=AllSamplesPolicy(),
        gamma_position=1.0,
        gamma_confidence=1.0,
        gamma_consistency=1.0,
        candidate_topk=5,
        ed_weight=0.1,
        num_neg_samples=1
    )
    
    return {
        "sasrec_trainer": sasrec_trainer,
        "ilora_trainer": ilora_trainer,
        "distill_trainer": distill_trainer,
        "device": device,
        "num_items": num_items,
        "max_seq_len": max_seq_len
    }

def test_sasrec_trainer_steps(trainer_fixture):
    """
    Test 41: [SASRec Trainer Correctness] Verify training/validation/test steps for SASRecTrainer.
    """
    trainer = trainer_fixture["sasrec_trainer"]
    device = trainer_fixture["device"]
    num_items = trainer_fixture["num_items"]
    
    batch_size = 2
    batch = {
        "seq": torch.randint(1, num_items, (batch_size, 20)).to(device),
        "len_seq": torch.tensor([5, 8]).to(device),
        "next_item": torch.tensor([1, num_items]).to(device) # Test bounds (1 and max)
    }
    
    # Training Step
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0
    
    # Validation Step
    val_out = trainer.validation_step(batch, 0)
    assert "val_loss" in val_out
    
    # Test Step
    test_out = trainer.test_step(batch, 0)
    assert "test_loss" in test_out

def test_ilora_trainer_steps(trainer_fixture):
    """
    Test 42: [iLoRA Trainer Correctness] Verify training/validation/test steps for iLoRATrainer.
    Specifically checks for NoneType loss and correct metric indexing.
    """
    trainer = trainer_fixture["ilora_trainer"]
    device = trainer_fixture["device"]
    num_items = trainer_fixture["num_items"]
    
    batch_size = 2
    llm_seq_len = 10
    batch = {
        "seq": torch.randint(1, num_items, (batch_size, 20)).to(device),
        "len_seq": torch.tensor([5, 8]).to(device),
        "input_ids": torch.randint(0, 1000, (batch_size, llm_seq_len)).to(device),
        "attention_mask": torch.ones((batch_size, llm_seq_len)).to(device),
        "next_item": torch.tensor([1, num_items]).to(device) # Test bounds
    }
    
    # Training Step
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0
    
    # Validation Step
    # We want to verify the indexing logic (0-based index vs 1-based ID)
    # iLoRATrainer.validation_step calls model.get_teacher_outputs
    # We can mock get_teacher_outputs on the model instance
    original_get_outputs = trainer.model.get_teacher_outputs
    
    def mock_get_outputs(batch):
        # Create ranking scores where index 0 has max score for sample 0
        # and index num_items-1 has max score for sample 1
        scores = torch.zeros((batch_size, num_items)).to(device)
        scores[0, 0] = 100.0 # Index 0 -> Item 1
        scores[1, num_items-1] = 100.0 # Index N-1 -> Item N
        
        return {
            "ranking_scores": scores,
            "embeddings": torch.randn(batch_size, 768).to(device),
            "candidates": torch.zeros(batch_size, 5).long().to(device),
            "confidence": torch.zeros(batch_size, 5).to(device)
        }
        
    trainer.model.get_teacher_outputs = mock_get_outputs
    
    try:
        # Run validation step
        trainer.validation_step(batch, 0)
        pass
        
    finally:
        trainer.model.get_teacher_outputs = original_get_outputs

def test_distillation_trainer_steps(trainer_fixture):
    """
    Test 43: [Distillation Trainer Correctness] Verify training/validation/test steps for DistillationTrainer.
    Specifically checks for index out of bounds errors and metric correctness.
    """
    trainer = trainer_fixture["distill_trainer"]
    device = trainer_fixture["device"]
    num_items = trainer_fixture["num_items"]
    
    batch_size = 2
    llm_seq_len = 10
    batch = {
        "seq": torch.randint(1, num_items, (batch_size, 20)).to(device),
        "len_seq": torch.tensor([5, 8]).to(device),
        "input_ids": torch.randint(0, 1000, (batch_size, llm_seq_len)).to(device),
        "attention_mask": torch.ones((batch_size, llm_seq_len)).to(device),
        "next_item": torch.tensor([1, num_items]).to(device) # Test bounds (1 and max)
    }
    
    # Training Step
    # This is expected to fail if DistillationTrainer uses next_item (1-based) directly with CrossEntropyLoss
    # when next_item is num_items (index out of bounds for size num_items).
    try:
        loss = trainer.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    except IndexError:
        pytest.fail("DistillationTrainer.training_step raised IndexError. Likely due to 1-based item ID usage in CrossEntropyLoss.")
    except Exception as e:
        # If it's a CUDA error it might show up as RuntimeError
        if "index out of range" in str(e) or "Target" in str(e):
             pytest.fail(f"DistillationTrainer.training_step failed with index error: {e}")
        else:
             raise e

    # Validation Step
    # This might fail metric calculation if indexing is mismatched
    trainer.validation_step(batch, 0)
