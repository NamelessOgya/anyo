import torch
import torch.nn as nn
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import pytest
from unittest.mock import MagicMock
from omegaconf import OmegaConf

def test_hidden_states_access():
    print("\n--- Starting Hidden States Access Test ---")
    
    # Create Dummy Config
    cfg = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 2,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "hidden_size": 32,
            "dropout_rate": 0.1,
            "initialize_item_embeddings_semantically": False
        },
        "train": {
            "precision": "32"
        }
    })
    
    num_items = 5
    max_seq_len = 10
    item_id_to_name = {i: f"Item {i}" for i in range(1, num_items + 1)}
    
    # Mock Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    # Create Model
    print("Creating teacher model...")
    model = create_teacher_model(
        cfg,
        llm_tokenizer=tokenizer,
        num_items=num_items,
        max_seq_len=max_seq_len,
        item_id_to_name=item_id_to_name,
        padding_item_id=0,
        candidate_topk=5
    )
    
    # Create Dummy Input
    input_ids = torch.tensor([[2, 100, 50266]], dtype=torch.long) # Dummy input
    attention_mask = torch.ones_like(input_ids)
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Test _get_llm_outputs (called by forward)
    print("Testing forward...")
    outputs = model(batch)
    
    assert hasattr(outputs, "hidden_states"), "Outputs should have hidden_states"
    assert outputs.hidden_states is not None, "hidden_states should not be None"
    print("Forward pass returned hidden_states.")
    
    # Test get_teacher_outputs
    print("Testing get_teacher_outputs...")
    teacher_outputs = model.get_teacher_outputs(batch)
    
    assert "embeddings" in teacher_outputs
    assert teacher_outputs["embeddings"].shape == (1, 768) # OPT-125m hidden size
    print("get_teacher_outputs passed.")
    
    print("Hidden States Access Test Passed.")

if __name__ == "__main__":
    test_hidden_states_access()
