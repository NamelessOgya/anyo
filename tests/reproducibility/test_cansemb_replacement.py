
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from omegaconf import OmegaConf
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector

class MockRecModel(nn.Module):
    def __init__(self, hidden_size, num_items):
        super().__init__()
        self.hidden_size = hidden_size
        # +1 for padding
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size)
        
    def get_full_sequence_representations(self, item_seq, item_seq_len):
        batch_size, seq_len = item_seq.shape
        return torch.randn(batch_size, seq_len, self.hidden_size, device=item_seq.device)

    def _get_last_item_representation(self, item_seq, item_seq_len):
        batch_size = item_seq.shape[0]
        return torch.randn(batch_size, self.hidden_size, device=item_seq.device)

@pytest.fixture
def dummy_ilora_model():
    # Setup minimal config
    hidden_size = 32
    num_items = 100
    llm_hidden_size = 768 # Match default OPT-125m
    
    # Mock LLM
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    # config.hidden_size = llm_hidden_size # Already 768
    config.num_hidden_layers = 2
    # config.num_attention_heads = 12 # Already 12
    llm = AutoModelForCausalLM.from_config(config)
    
    # Mock Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))
    
    # Mock Rec Model
    rec_model = MockRecModel(hidden_size, num_items)
    
    # Mock Projector
    projector = MLPProjector(hidden_size, llm_hidden_size, hidden_size, 0.1)
    
    # Create iLoRAModel
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
    return model

def test_cansemb_replacement(dummy_ilora_model):
    model = dummy_ilora_model
    tokenizer = model.tokenizer
    
    # Create a dummy batch
    # Prompt: "Predict next: [CansEmb]"
    cans_token_id = tokenizer.convert_tokens_to_ids("[CansEmb]")
    prompt_text = "Predict next: [CansEmb]"
    inputs = tokenizer(prompt_text, return_tensors="pt")
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Dummy candidates (batch_size=1, num_candidates=3)
    candidates = torch.tensor([[1, 2, 3]]) 
    len_cans = torch.tensor([3])
    
    # Dummy sequence data (needed for other parts of forward pass)
    seq = torch.tensor([[4, 5]])
    len_seq = torch.tensor([2])
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": seq,
        "len_seq": len_seq,
        "cans": candidates,
        "len_cans": len_cans,
        "next_item": torch.tensor([1]) # Dummy
    }
    
    # We want to intercept the call to self.llm to check inputs_embeds
    # But since self.llm is a MoeLoraModel wrapper, we can inspect the hook or just check the output logic if we could.
    # A better way is to spy on the internal method or check if the embedding at [CansEmb] position 
    # is different from the standard token embedding.
    
    # 1. Get standard token embedding for [CansEmb]
    base_embeddings = model.llm.get_input_embeddings()(input_ids)
    cans_pos = (input_ids == cans_token_id).nonzero(as_tuple=True)
    original_cans_embedding = base_embeddings[cans_pos]
    
    # 2. Run forward pass (which calls _get_llm_outputs)
    # We need to mock self.llm to capture the inputs_embeds passed to it
    original_llm_forward = model.llm.forward
    captured_inputs_embeds = []
    
    def mock_forward(inputs_embeds, **kwargs):
        captured_inputs_embeds.append(inputs_embeds)
        return original_llm_forward(inputs_embeds=inputs_embeds, **kwargs)
        
    model.llm.forward = mock_forward
    
    try:
        model._get_llm_outputs(batch)
    finally:
        model.llm.forward = original_llm_forward # Restore
        
    assert len(captured_inputs_embeds) > 0
    modified_embeddings = captured_inputs_embeds[0]
    
    # 3. Check if the embedding at [CansEmb] position has changed
    new_cans_embedding = modified_embeddings[cans_pos]
    
    # The new embedding should be different from the original token embedding
    assert not torch.allclose(original_cans_embedding, new_cans_embedding), \
        "[CansEmb] embedding was not replaced!"
        
    # 4. Verify that the new embedding comes from the candidate item
    # We can manually calculate what it should be
    # The first candidate is ID 1
    # encode_items calls rec_model.item_embeddings(1) then projector
    # But encode_items takes full sequence representations for history, 
    # wait, encode_items implementation in iLoRAModel uses projector on full_sequence_representations.
    # Let's check iLoRAModel.encode_items again.
    
    # In iLoRAModel.encode_items:
    # return self.projector(full_sequence_representations)
    
    # But wait, for candidates, we pass item IDs to encode_items?
    # In iLoRAModel._get_llm_outputs:
    # cans_item_embeds = self.encode_items(batch["cans"])
    
    # If batch["cans"] is passed (which are IDs), then full_sequence_representations in encode_items will be IDs?
    # No, encode_items expects representations.
    # Let's check iLoRAModel.encode_items signature.
    # def encode_items(self, full_sequence_representations: torch.Tensor) -> torch.Tensor:
    #    return self.projector(full_sequence_representations)
    
    # Ah, if we pass IDs to encode_items, it will fail if it expects tensors.
    # BUT, in the fix I implemented:
    # cans_item_embeds = self.encode_items(batch["cans"])
    
    # If batch["cans"] is a tensor of IDs, then self.projector(IDs) will fail because projector expects float tensor input (embeddings).
    # The projector is an MLP. It does NOT include the embedding layer.
    # The embedding lookup happens BEFORE projector in the reference code?
    # Let's check reference model_interface.py:
    # def encode_items(self, seq):
    #     if self.hparams.rec_embed=="SASRec":
    #         item_rec_embs=self.rec_model.cacu_x(seq) ...
    #     item_txt_embs=self.projector(item_rec_embs)
    
    # My iLoRAModel.encode_items:
    # def encode_items(self, full_sequence_representations: torch.Tensor) -> torch.Tensor:
    #    return self.projector(full_sequence_representations)
    
    # ERROR! My encode_items assumes input is already embeddings/representations.
    # But for candidates, I am passing IDs!
    # I need to fix iLoRAModel to handle ID-to-Embedding lookup for candidates!
    
    pass
