import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector
from src.student.models import SASRec

def test_sasrec_device_placement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda")
    
    # 1. Setup Models
    num_items = 100
    hidden_size = 32
    max_seq_len = 50
    
    rec_model = SASRec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=max_seq_len
    )
    
    # Move to GPU
    rec_model.to(device)
    
    # Check parameters
    assert rec_model.item_embeddings.weight.device.type == "cuda", "SASRec embeddings should be on CUDA"
    for param in rec_model.parameters():
        assert param.device.type == "cuda", f"Parameter {param} should be on CUDA"
        
    # 2. Run Forward
    seq = torch.randint(1, num_items, (4, max_seq_len)).to(device)
    len_seq = torch.full((4,), max_seq_len).to(device)
    
    output = rec_model.get_full_sequence_representations(seq, len_seq)
    
    assert output.device.type == "cuda", "SASRec output should be on CUDA"

def test_ilora_integration_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = torch.device("cuda")
    
    # Setup minimal iLoRAModel
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    config.hidden_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    
    llm = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[HistoryEmb]']})
    llm.resize_token_embeddings(len(tokenizer))
    
    num_items = 100
    hidden_size = 32
    
    rec_model = SASRec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=50
    )
    
    projector = MLPProjector(hidden_size, config.hidden_size, hidden_size, 0.1)
    
    model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_items=num_items,
        num_lora_experts=4,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        hidden_size=hidden_size,
        dropout_rate=0.1,
        rec_model=rec_model,
        projector=projector,
        candidate_topk=10,
        item_id_to_name={i: f"Item {i}" for i in range(num_items+1)},
        padding_item_id=0,
        llm_dtype=torch.float32
    )
    
    model.to(device)
    
    # Check rec_model inside iLoRA
    assert model.rec_model.item_embeddings.weight.device.type == "cuda"
    
    # Create dummy batch on GPU
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones((batch_size, seq_len)).to(device)
    seq = torch.randint(1, num_items, (batch_size, 50)).to(device)
    len_seq = torch.full((batch_size,), 50).to(device)
    next_item = torch.randint(1, num_items, (batch_size,)).to(device)
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": seq,
        "len_seq": len_seq,
        "next_item": next_item
    }
    
    # Run forward
    outputs = model.get_teacher_outputs(batch)
    
    assert outputs["ranking_scores"].device.type == "cuda"
    assert outputs["embeddings"].device.type == "cuda"
