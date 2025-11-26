import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector

class DummyRecModel(nn.Module):
    def __init__(self, hidden_size, num_items):
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size)
        self.hidden_size = hidden_size
    
    def get_full_sequence_representations(self, seq, len_seq):
        return torch.randn(seq.shape[0], seq.shape[1], self.hidden_size, device=seq.device)
    
    def _get_last_item_representation(self, seq, len_seq):
        return torch.randn(seq.shape[0], self.hidden_size, device=seq.device)

@pytest.fixture
def ilora_model_setup():
    # Use a tiny config for speed
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    config.hidden_size = 64 # Tiny hidden size
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.ffn_dim = 256
    
    # Mock LLM creation to avoid downloading/loading heavy weights
    llm = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m") # Tokenizer is fast enough
    
    num_items = 100
    hidden_size = 32
    
    rec_model = DummyRecModel(hidden_size, num_items)
    # Freeze rec_model as expected in factory
    for param in rec_model.parameters():
        param.requires_grad = False
        
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
        item_id_to_name={i: f"Item {i}" for i in range(num_items + 1)},
        padding_item_id=0,
        llm_dtype=torch.float32
    )
    return model

def test_trainable_parameters_ratio(ilora_model_setup):
    model = ilora_model_setup
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    ratio = trainable_params / total_params
    print(f"Total Params: {total_params}")
    print(f"Trainable Params: {trainable_params}")
    print(f"Ratio: {ratio:.4f}")
    
    # Expect ratio to be small (e.g., < 10% or even < 1% depending on model size)
    # With tiny model, ratio might be higher, but let's check it's not 100%
    assert ratio < 0.5, f"Trainable parameter ratio {ratio} is too high! Base LLM might not be frozen."
    
    # For a real large model, this would be < 0.05 usually.
    # Since we use a tiny config (H=64), the LoRA overhead is relatively larger.
    # But checking specific layers is more robust.

def test_layer_freeze_status(ilora_model_setup):
    model = ilora_model_setup
    
    # 1. Base LLM weights (e.g., layernorm, embeddings) should be frozen
    # Note: MoeLoraModel wraps layers.
    # Check a non-LoRA layer, e.g., layer norm or embedding
    assert not model.llm.model.model.decoder.embed_tokens.weight.requires_grad, "LLM Embeddings should be frozen"
    assert not model.llm.model.model.decoder.layers[0].self_attn.k_proj.weight.requires_grad, "Non-target LLM layers should be frozen"
    
    # 2. Rec Model should be frozen
    for param in model.rec_model.parameters():
        assert not param.requires_grad, "Rec Model should be frozen"
        
    # 3. LoRA Adapters should be trainable
    # Find a LoRA layer
    found_lora = False
    for name, module in model.llm.named_modules():
        if hasattr(module, "lora_A"):
            found_lora = True
            # Check adapter weights
            for adapter_name in module.lora_A:
                assert module.lora_A[adapter_name].weight.requires_grad, f"LoRA A {adapter_name} should be trainable"
                assert module.lora_B[adapter_name].weight.requires_grad, f"LoRA B {adapter_name} should be trainable"
    assert found_lora, "No LoRA layers found in the model"

    # 4. Gating Network should be trainable
    for param in model.gating_network.parameters():
        assert param.requires_grad, "Gating Network should be trainable"

    # 5. Item Prediction Head should be trainable
    for param in model.item_prediction_head.parameters():
        assert param.requires_grad, "Item Prediction Head should be trainable"
        
    # 6. Projector should be trainable
    for param in model.projector.parameters():
        assert param.requires_grad, "Projector should be trainable"
