import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch
from src.teacher.trainer_ilora import iLoRATrainer
from src.teacher.ilora_model import iLoRAModel

class MockLLM:
    def __init__(self, vocab_size, hidden_size):
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = vocab_size
        self.embeddings = torch.randn(vocab_size, hidden_size)
        
    def get_input_embeddings(self):
        # Return a function that mimics embedding lookup
        def embedding_lookup(indices):
            return self.embeddings[indices]
        return embedding_lookup
        
    def __call__(self, *args, **kwargs):
        # Return dummy output
        batch_size = kwargs.get('inputs_embeds', torch.zeros(1, 1, 1)).shape[0]
        seq_len = kwargs.get('inputs_embeds', torch.zeros(1, 1, 1)).shape[1]
        last_hidden_state = torch.randn(batch_size, seq_len, self.config.hidden_size)
        return MagicMock(last_hidden_state=last_hidden_state)

class MockiLoRAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.original_vocab_size = 1000
        self.num_items = 100
        self.hidden_size = 32
        self.llm = MockLLM(self.original_vocab_size + self.num_items + 1, self.hidden_size)
        self.use_item_embeddings_head = False # Not used in E4SRec logic directly in trainer, but logic changed
        
    def forward(self, batch):
        return self.llm(inputs_embeds=batch.get('input_ids')) # Dummy

@pytest.fixture
def mock_trainer():
    model = MockiLoRAModel()
    trainer = iLoRATrainer(
        ilora_model=model,
        num_items=100,
        learning_rate=1e-4,
        weight_decay=0.01,
        metrics_k=10,
        item_id_to_name={},
        distill_lambda=0.0
    )
    trainer.log = MagicMock()
    return trainer

def test_training_step_e4srec(mock_trainer):
    # Setup batch
    batch_size = 4
    seq_len = 10
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "next_item": torch.tensor([1, 2, 3, 4]) # 1-based item IDs
    }
    
    # Run training_step
    loss = mock_trainer.training_step(batch, batch_idx=0)
    
    # Verify
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0
    
    # Check if log was called
    mock_trainer.log.assert_called()

def test_training_step_sampled_softmax_logic(mock_trainer):
    # Verify that sampled softmax logic is using correct embeddings
    # We can mock F.cross_entropy to check inputs
    
    batch_size = 2
    seq_len = 5
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "next_item": torch.tensor([1, 2]) # Item 1, Item 2
    }
    
    with patch('torch.nn.functional.cross_entropy') as mock_ce:
        mock_ce.return_value = torch.tensor(0.5)
        
        mock_trainer.training_step(batch, batch_idx=0)
        
        # Check arguments to cross_entropy
        args, kwargs = mock_ce.call_args
        logits = args[0]
        targets = args[1]
        
        # Logits shape should be (Batch, NumSamples)
        # NumSamples is 2048 in code, but limited by num_items=100 in mock
        # Logic: num_negatives = num_samples - len(unique_targets)
        # If num_samples (2048) > num_total_items (100), it might sample duplicates or clamp?
        # Code: torch.randint(min, max, (num_negatives,))
        # It samples 2048 negatives regardless of num_items size?
        # Yes, standard implementation samples with replacement if needed or just samples.
        # But wait, `sampled_indices = torch.unique(sampled_indices)`
        # So if num_items is small (100), unique samples will be at most 100.
        
        assert logits.shape[0] == batch_size
        assert logits.shape[1] <= 100 # Since we only have 100 items
        assert targets.shape[0] == batch_size
        
        # Verify targets are within range [0, logits.shape[1]-1]
        assert (targets >= 0).all()
        assert (targets < logits.shape[1]).all()
