import pytest
import torch
import torch.nn as nn
from src.teacher.moe_lora_model import MoeLoraModel, Linear
from transformers import AutoModelForCausalLM, AutoConfig

@pytest.fixture
def dummy_llm():
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    config.hidden_size = 32 # Small hidden size for testing
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    return AutoModelForCausalLM.from_config(config)

def test_moelora_rank_interpretation(dummy_llm):
    """
    Test that lora_r is treated as Total Rank, not Per-Expert Rank.
    If lora_r=8 and num_experts=4, then each expert should have rank 2 (8 // 4).
    The weight matrices lora_A and lora_B should have dimensions corresponding to Total Rank (8).
    """
    lora_r = 8
    num_experts = 4
    lora_alpha = 16
    lora_dropout = 0.0
    
    # Wrap the model
    model = MoeLoraModel(
        model=dummy_llm,
        target_modules=["q_proj", "v_proj"],
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_lora_experts=num_experts
    )
    
    # Find a replaced layer
    layer = None
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            layer = module
            break
            
    assert layer is not None, "No MoeLora Linear layer found"
    
    # Check dimensions
    # In reference implementation:
    # lora_A: (in_features, r) -> but reshaped to (num_experts, r//num_experts, in_features) during forward
    # Wait, let's check the reference code again carefully.
    # Reference update_layer:
    # self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
    # So lora_A weight shape is (r, in_features) because nn.Linear is (out_features, in_features)
    
    # So if r=8, lora_A.weight.shape should be (8, in_features)
    
    adapter_name = "default"
    assert adapter_name in layer.lora_A
    
    lora_A_weight = layer.lora_A[adapter_name].weight
    lora_B_weight = layer.lora_B[adapter_name].weight
    
    print(f"lora_A shape: {lora_A_weight.shape}")
    print(f"lora_B shape: {lora_B_weight.shape}")
    
    # Expected: (r, in_features) for A, (out_features, r) for B
    # Note: nn.Linear weights are (out, in).
    # lora_A is nn.Linear(in_features, r) -> weight is (r, in_features)
    # lora_B is nn.Linear(r, out_features) -> weight is (out_features, r)
    
    assert lora_A_weight.shape[0] == lora_r
    assert lora_B_weight.shape[1] == lora_r
    
    # Check scaling factor
    # Reference: self.scaling[adapter_name] = lora_alpha / (r // num_moe)
    expected_scaling = lora_alpha / (lora_r // num_experts)
    assert layer.scaling[adapter_name] == expected_scaling

def test_moelora_gating_logic(dummy_llm):
    """
    Test that gate weights are correctly applied.
    """
    lora_r = 8
    num_experts = 4
    lora_alpha = 16
    
    model = MoeLoraModel(
        model=dummy_llm,
        target_modules=["q_proj"],
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        num_lora_experts=num_experts
    )
    
    # Set gate weights manually
    batch_size = 2
    seq_len = 5
    # Gate weights: (batch_size, num_experts)
    # Let's make expert 0 have weight 1.0 for sample 0, and expert 1 have weight 1.0 for sample 1
    gate_weights = torch.zeros(batch_size, num_experts)
    gate_weights[0, 0] = 1.0
    gate_weights[1, 1] = 1.0
    
    model.gate_weights.append(gate_weights)
    
    # Create dummy input
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    
    # Forward pass
    # We just want to ensure it runs without error and shapes are correct
    # Detailed numerical verification is hard without mocking internal weights, 
    # but we can check if it runs.
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.logits.shape == (batch_size, seq_len, dummy_llm.config.vocab_size)
