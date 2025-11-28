import torch
import torch.nn as nn
from omegaconf import OmegaConf
from src.teacher.factory import create_teacher_model
from src.teacher.ilora_model import iLoRAModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def test_teacher_embedding_head_and_unfreeze():
    # 1. Setup Dummy Config
    cfg = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 2,
            "lora_r": 4,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "hidden_size": 32,
            "dropout_rate": 0.0,
            "rec_model_checkpoint_path": "dummy_ckpt.pth",
            "use_flash_attention": False,
            "use_qlora": False,
            "use_gradient_checkpointing": False,
            "use_torch_compile": False
        },
        "student": {
            "hidden_size": 32,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.0,
            "max_seq_len": 20,
        },
        "train": {
            "precision": "32"
        }
    })

    # 2. Create Dummy Checkpoint
    ckpt_path = os.path.abspath("dummy_ckpt.pth")
    cfg.teacher.rec_model_checkpoint_path = ckpt_path
    
    dummy_state = {
        'state_dict': {
            'model.item_embeddings.weight': torch.randn(101, 32), # 100 items + 1 padding
            'model.position_embeddings.weight': torch.randn(20, 32),
            'model.emb_dropout.weight': torch.randn(32), # Dummy
            # Add other layers if needed by SASRec load
        }
    }
    # SASRec strict loading might fail if we don't provide all keys.
    # But factory.py does: rec_model.load_state_dict(new_state_dict) (default strict=True)
    # We need to populate enough keys or mock SASRec.
    
    # Let's just mock the SASRec class inside factory? No, we want to test factory logic.
    # We can use a real SASRec and save its state.
    from src.student.models import SASRec
    rec_model = SASRec(num_items=100, hidden_size=32, num_heads=2, num_layers=2, dropout_rate=0.0, max_seq_len=20, padding_item_id=0)
    
    # Wrap in 'model.' prefix as expected by factory
    save_dict = {'state_dict': {f'model.{k}': v for k, v in rec_model.state_dict().items()}}
    torch.save(save_dict, ckpt_path)

    # 3. Create Teacher Model
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    
    try:
        model = create_teacher_model(
            cfg,
            llm_tokenizer=tokenizer,
            num_items=100,
            max_seq_len=20,
            item_id_to_name={i: str(i) for i in range(101)},
            padding_item_id=0,
            candidate_topk=5
        )
        
        # 4. Verify Item Embeddings are Unfrozen
        assert model.rec_model.item_embeddings.weight.requires_grad == True, "Item embeddings should be trainable"
        
        # Verify other parameters are frozen
        for name, param in model.rec_model.named_parameters():
            if "item_embeddings" not in name:
                assert param.requires_grad == False, f"Parameter {name} should be frozen"

        # 5. Verify Forward Pass (Logits Shape)
        # Create dummy batch
        # SASRec expects seq to be max_seq_len (20)
        seq = torch.zeros(2, 20, dtype=torch.long)
        seq[:, -5:] = torch.randint(1, 100, (2, 5)) # Last 5 items are valid
        
        batch = {
            "seq": seq,
            "len_seq": torch.tensor([5, 5]),
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "next_item": torch.tensor([1, 2])
        }
        
        outputs = model.get_teacher_outputs(batch)
        logits = outputs["ranking_scores"]
        
        # Shape should be (Batch, Num_Items + 1) or (Batch, Num_Items)?
        # iLoRAModel logic: 
        # all_items_indices = torch.arange(self.rec_model.item_embeddings.num_embeddings)
        # So it returns logits for ALL embeddings (including padding).
        # Shape: (2, 101)
        assert logits.shape == (2, 101)
        
        # 6. Verify Gradients Flow to Item Embeddings
        loss = logits.sum()
        loss.backward()
        
        assert model.rec_model.item_embeddings.weight.grad is not None, "Gradients should flow to item embeddings"
        assert model.rec_model.item_embeddings.weight.grad.abs().sum() > 0, "Gradients should be non-zero"

    finally:
        if os.path.exists("dummy_ckpt.pth"):
            os.remove("dummy_ckpt.pth")

if __name__ == "__main__":
    test_teacher_embedding_head_and_unfreeze()
