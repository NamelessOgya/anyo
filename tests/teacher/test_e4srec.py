import pytest
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
from unittest.mock import MagicMock, patch
from src.student.datamodule import TeacherTrainCollater
from src.teacher.ilora_model import iLoRAModel
from src.teacher.factory import create_teacher_model
from omegaconf import OmegaConf

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.additional_special_tokens_ids = [999] # [HistoryEmb]
        self.additional_special_tokens = ["[HistoryEmb]"]
    
    def __len__(self):
        return self.vocab_size
    
    def __call__(self, text, **kwargs):
        # Simple mock tokenization
        if isinstance(text, str):
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        elif isinstance(text, list):
            return {
                "input_ids": torch.tensor([[1, 2, 3]] * len(text)), 
                "attention_mask": torch.tensor([[1, 1, 1]] * len(text))
            }

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

def test_collater_e4srec_format(mock_tokenizer):
    # Setup
    max_seq_len = 10
    padding_item_id = 0
    id_to_name = {1: "Item1", 2: "Item2"}
    collater = TeacherTrainCollater(mock_tokenizer, max_seq_len, padding_item_id, id_to_name)
    
    # Mock tokenizer call for prefix/suffix
    collater.prefix_ids = torch.tensor([10, 11]) # "This user has watched "
    collater.suffix_ids = torch.tensor([12, 13]) # " in the previous..."
    collater.vocab_offset = 1000
    
    # Batch data
    batch = [{
        "seq_ids": [1, 2],
        "next_item_id": 3,
        "candidates": [4, 5],
        "history_str": "unused",
        "candidates_str": "unused"
    }]
    
    # Execute
    output = collater(batch)
    
    # Verify
    input_ids = output["input_ids"][0]
    labels = output["labels"][0]
    
    # Expected structure: [Prefix(2), Item1(1001), Item2(1002), Suffix(2), Target(1003)]
    # Length = 2 + 2 + 2 + 1 = 7
    
    expected_item_tokens = torch.tensor([1001, 1002])
    expected_target_token = torch.tensor([1003])
    
    # Check Item Tokens (indices 2, 3)
    assert torch.equal(input_ids[2:4], expected_item_tokens)
    
    # Check Target Token (last index)
    assert input_ids[-1] == expected_target_token
    
    # Check Labels
    # Should be -100 everywhere except last position
    assert torch.all(labels[:-1] == -100)
    assert labels[-1] == expected_target_token

def test_ilora_model_gating_logic():
    # Setup
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    config.vocab_size = 1000
    llm = AutoModel.from_config(config)
    tokenizer = MockTokenizer()
    
    num_items = 10
    original_vocab_size = 1000
    
    # Resize LLM embeddings
    llm.resize_token_embeddings(original_vocab_size + num_items + 1)
    
    model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_items=num_items,
        num_lora_experts=2,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        hidden_size=32,
        dropout_rate=0.1,
        candidate_topk=5,
        item_id_to_name={i: f"Item{i}" for i in range(1, num_items+1)},
        padding_item_id=0,
        llm_dtype=torch.float32,
        original_vocab_size=original_vocab_size
    )
    
    # Create dummy input
    # [Prefix(2), Item1(1001), Item2(1002), Suffix(2), Target(1003)]
    input_ids = torch.tensor([[10, 11, 1001, 1002, 12, 13, 1003]])
    attention_mask = torch.ones_like(input_ids)
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": torch.tensor([[1, 2]]),
        "len_seq": torch.tensor([2])
    }
    
    # Run forward
    outputs = model(batch)
    
    # Verify Gating Logic
    # The gating network should have received the average of embeddings for 1001 and 1002
    # We can't easily check the internal tensor without mocking, but we can check if it runs.
    # AutoModel output has last_hidden_state
    assert hasattr(outputs, "last_hidden_state")
    
    # Check if gate weights were generated
    assert len(model.llm.gate_weights) > 0
    assert model.llm.gate_weights[0].shape == (1, 2) # (Batch, NumExperts)

def test_ilora_model_scoring():
    # Setup (same as above)
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    config.vocab_size = 1000
    llm = AutoModel.from_config(config)
    tokenizer = MockTokenizer()
    num_items = 10
    original_vocab_size = 1000
    llm.resize_token_embeddings(original_vocab_size + num_items + 1)
    
    model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_items=num_items,
        num_lora_experts=2,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        hidden_size=32,
        dropout_rate=0.1,
        candidate_topk=5,
        item_id_to_name={i: f"Item{i}" for i in range(1, num_items+1)},
        padding_item_id=0,
        llm_dtype=torch.float32,
        original_vocab_size=original_vocab_size
    )
    
    input_ids = torch.tensor([[10, 11, 1001, 1002, 12, 13, 1003]])
    attention_mask = torch.ones_like(input_ids)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": torch.tensor([[1, 2]]),
        "len_seq": torch.tensor([2])
    }
    
    # Run get_teacher_outputs
    outputs = model.get_teacher_outputs(batch)
    
    # Verify Scores
    # Shape should be (Batch, NumItems)
    # Note: We are scoring against ALL items (1..num_items)
    # The implementation might include padding (0) in the score matrix but mask it?
    # Let's check the shape.
    # ranking_scores = last_hidden_state @ item_embeddings.T
    # item_embeddings shape is (num_items, H)
    # so scores shape is (Batch, num_items)
    
    ranking_scores = outputs["ranking_scores"]
    # Shape should be (Batch, NumItems + 1) to include padding at index 0
    assert ranking_scores.shape == (1, num_items + 1)
    
    # Check if padding item (Index 0) is masked
    assert ranking_scores[0, 0] == float('-inf')
    
    # Check if valid items have finite scores
    assert torch.isfinite(ranking_scores[0, 1]).all()
