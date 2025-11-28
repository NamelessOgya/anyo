
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector

class MockRecModel(nn.Module):
    def __init__(self, hidden_size, num_items):
        super().__init__()
        self.hidden_size = hidden_size
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size)
        
    def get_full_sequence_representations(self, item_seq, item_seq_len):
        # Return embeddings corresponding to item_seq
        # This allows us to verify which item was used for replacement
        return self.item_embeddings(item_seq)

    def _get_last_item_representation(self, item_seq, item_seq_len):
        batch_size = item_seq.shape[0]
        return torch.randn(batch_size, self.hidden_size, device=item_seq.device)

@pytest.fixture
def model_setup():
    hidden_size = 32
    num_items = 10
    llm_hidden_size = 32 # Small for testing
    
    # Mock LLM
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    config.hidden_size = llm_hidden_size
    config.word_embed_proj_dim = llm_hidden_size # Ensure embedding dim is 32
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    llm = AutoModelForCausalLM.from_config(config)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))
    
    rec_model = MockRecModel(hidden_size, num_items)
    
    # Identity projector for easy verification
    class MockProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(1, 1)) # Dummy to satisfy dtype check
        def forward(self, x):
            return x # Identity
            
    projector = MockProjector()
    
    model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_items=num_items,
        num_lora_experts=2,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        hidden_size=hidden_size,
        dropout_rate=0.0,
        rec_model=rec_model,
        projector=projector,
        candidate_topk=5,
        item_id_to_name={i: f"Item_{i}" for i in range(1, num_items + 1)},
        padding_item_id=0,
        llm_dtype=torch.float32
    )
    return model, rec_model, projector

def test_history_replacement_logic(model_setup):
    model, rec_model, projector = model_setup
    tokenizer = model.tokenizer
    
    # Case 1: 3 placeholders, 3 valid items -> Replace all 3
    # Items: [1, 2, 3]
    # Prompt: "A [HistoryEmb] B [HistoryEmb] C [HistoryEmb]"
    
    his_token_id = tokenizer.convert_tokens_to_ids("[HistoryEmb]")
    input_ids = torch.tensor([[10, his_token_id, 11, his_token_id, 12, his_token_id]])
    attention_mask = torch.ones_like(input_ids)
    
    seq = torch.tensor([[1, 2, 3]])
    len_seq = torch.tensor([3])
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": seq,
        "len_seq": len_seq
    }
    
    # Capture inputs to LLM
    original_llm_forward = model.llm.forward
    captured_inputs = []
    def mock_forward(inputs_embeds, **kwargs):
        captured_inputs.append(inputs_embeds)
        return original_llm_forward(inputs_embeds=inputs_embeds, **kwargs)
    model.llm.forward = mock_forward
    
    model._get_llm_outputs(batch)
    model.llm.forward = original_llm_forward
    
    inputs_embeds = captured_inputs[0]
    
    # Expected embeddings
    # Item 1 -> Placeholder 1 (index 1)
    # Item 2 -> Placeholder 2 (index 3)
    # Item 3 -> Placeholder 3 (index 5)
    
    # Calculate expected projected embeddings
    item_embs = rec_model.item_embeddings(seq) # (1, 3, 32)
    proj_embs = model.projector(item_embs) # (1, 3, 32)
    
    assert torch.allclose(inputs_embeds[0, 1], proj_embs[0, 0])
    assert torch.allclose(inputs_embeds[0, 3], proj_embs[0, 1])
    assert torch.allclose(inputs_embeds[0, 5], proj_embs[0, 2])

def test_history_replacement_truncation(model_setup):
    model, rec_model, projector = model_setup
    tokenizer = model.tokenizer
    
    # Case 2: 3 placeholders, 2 valid items -> Replace last 2 placeholders with last 2 items
    # Items: [0, 2, 3] (0 is padding, valid are 2, 3)
    # Prompt: "A [HistoryEmb] B [HistoryEmb] C [HistoryEmb]"
    
    his_token_id = tokenizer.convert_tokens_to_ids("[HistoryEmb]")
    input_ids = torch.tensor([[10, his_token_id, 11, his_token_id, 12, his_token_id]])
    attention_mask = torch.ones_like(input_ids)
    
    seq = torch.tensor([[0, 2, 3]]) # 0 is padding
    len_seq = torch.tensor([2])
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": seq,
        "len_seq": len_seq
    }
    
    # Capture inputs
    original_llm_forward = model.llm.forward
    captured_inputs = []
    def mock_forward(inputs_embeds, **kwargs):
        captured_inputs.append(inputs_embeds)
        return original_llm_forward(inputs_embeds=inputs_embeds, **kwargs)
    model.llm.forward = mock_forward
    
    model._get_llm_outputs(batch)
    model.llm.forward = original_llm_forward
    
    inputs_embeds = captured_inputs[0]
    
    # Expected:
    # Placeholder 1 (index 1) -> NOT REPLACED (should be original token embedding)
    # Placeholder 2 (index 3) -> Item 2
    # Placeholder 3 (index 5) -> Item 3
    
    item_embs = rec_model.item_embeddings(torch.tensor([[2, 3]])) # (1, 2, 32)
    proj_embs = model.projector(item_embs)
    
    # Check original embedding at index 1
    base_embs = model.llm.get_input_embeddings()(input_ids)
    assert torch.allclose(inputs_embeds[0, 1], base_embs[0, 1])
    
    # Check replaced embeddings
    assert torch.allclose(inputs_embeds[0, 3], proj_embs[0, 0])
    assert torch.allclose(inputs_embeds[0, 5], proj_embs[0, 1])

def test_cans_replacement(model_setup):
    model, rec_model, projector = model_setup
    tokenizer = model.tokenizer
    
    # Case: 2 placeholders, 2 candidates
    cans_token_id = tokenizer.convert_tokens_to_ids("[CansEmb]")
    input_ids = torch.tensor([[20, cans_token_id, 21, cans_token_id]])
    attention_mask = torch.ones_like(input_ids)
    
    seq = torch.tensor([[1]]) # Dummy
    len_seq = torch.tensor([1])
    
    candidates = torch.tensor([[4, 5]])
    len_cans = torch.tensor([2])
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": seq,
        "len_seq": len_seq,
        "cans": candidates,
        "len_cans": len_cans,
        "next_item": torch.tensor([1])
    }
    
    # Capture inputs
    original_llm_forward = model.llm.forward
    captured_inputs = []
    def mock_forward(inputs_embeds, **kwargs):
        captured_inputs.append(inputs_embeds)
        return original_llm_forward(inputs_embeds=inputs_embeds, **kwargs)
    model.llm.forward = mock_forward
    
    model._get_llm_outputs(batch)
    model.llm.forward = original_llm_forward
    
    inputs_embeds = captured_inputs[0]
    
    # Expected:
    # Placeholder 1 (index 1) -> Candidate 4
    # Placeholder 2 (index 3) -> Candidate 5
    
    cand_embs = rec_model.item_embeddings(candidates)
    proj_cand_embs = model.projector(cand_embs)
    
    assert torch.allclose(inputs_embeds[0, 1], proj_cand_embs[0, 0])
    assert torch.allclose(inputs_embeds[0, 3], proj_cand_embs[0, 1])
