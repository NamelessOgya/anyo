import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
# Add cmd/run_bigrec_posttraining to path for direct import
sys.path.append(str(Path(__file__).resolve().parents[2] / "cmd" / "run_bigrec_posttraining"))

from ensemble_model import EnsembleBigRecSASRec, AlphaNetwork
from run import EnsembleDataset, EnsembleCollator

class MockSASRec(nn.Module):
    def __init__(self, hidden_size, num_items):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_items = num_items
        
    def forward(self, seq, seq_len):
        batch_size = seq.shape[0]
        # Return dummy embedding
        return torch.randn(batch_size, self.hidden_size)
    
    def predict(self, seq, seq_len):
        batch_size = seq.shape[0]
        # Return dummy logits (including padding 0)
        return torch.randn(batch_size, self.num_items + 1)

def test_alpha_network():
    input_dim = 64
    net = AlphaNetwork(input_dim)
    
    batch_size = 10
    x = torch.randn(batch_size, input_dim)
    output = net(x)
    
    assert output.shape == (batch_size, 1)
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)

def test_ensemble_dataset_collator():
    # Mock data
    original_dataset = [
        {"seq": [1, 2], "len_seq": 2, "next_item": 3},
        {"seq": [4, 5, 6], "len_seq": 3, "next_item": 7}
    ]
    bigrec_embs = torch.randn(2, 32) # 2 samples, 32 dim
    
    dataset = EnsembleDataset(original_dataset, bigrec_embs)
    assert len(dataset) == 2
    
    item0 = dataset[0]
    assert item0["original"] == original_dataset[0]
    assert torch.equal(item0["bigrec_emb"], bigrec_embs[0])
    
    # Mock student collator
    def mock_student_collator(batch):
        return {
            "seq": torch.tensor([b["seq"] for b in batch]), # Simplified, assumes same length or padding handled
            "len_seq": torch.tensor([b["len_seq"] for b in batch]),
            "next_item": torch.tensor([b["next_item"] for b in batch])
        }
        
    collator = EnsembleCollator(mock_student_collator)
    
    batch = [dataset[0], dataset[1]]
    # Fix mock collator input for list of dicts
    # The dataset returns {"original": ..., "bigrec_emb": ...}
    # The collator separates them.
    
    # Actually, let's just run it.
    # But wait, mock_student_collator expects list of "original" dicts.
    # EnsembleCollator extracts "original" and passes to student_collator.
    # So mock_student_collator receives [original_dataset[0], original_dataset[1]]
    
    # We need to handle padding in mock_student_collator if seqs differ in length
    # For this test, let's make them same length or just handle list
    original_dataset[0]["seq"] = [1, 2, 0] # Pad manually
    original_dataset[0]["len_seq"] = 2
    
    batch_out = collator(batch)
    
    assert "bigrec_emb" in batch_out
    assert batch_out["bigrec_emb"].shape == (2, 32)
    assert "seq" in batch_out
    assert batch_out["seq"].shape == (2, 3)

def test_ensemble_model():
    hidden_size = 64
    num_items = 100
    emb_dim = 32
    
    sasrec = MockSASRec(hidden_size, num_items)
    alpha_net = AlphaNetwork(hidden_size)
    item_embeddings = torch.randn(num_items + 1, emb_dim)
    
    model = EnsembleBigRecSASRec(
        sasrec_model=sasrec,
        alpha_net=alpha_net,
        item_embeddings=item_embeddings,
        popularity_scores=None,
        popularity_lambda=0.0
    )
    
    batch_size = 5
    seq = torch.randint(1, num_items, (batch_size, 10))
    seq_len = torch.full((batch_size,), 10)
    bigrec_emb = torch.randn(batch_size, emb_dim)
    target = torch.randint(1, num_items, (batch_size,))
    
    # Test Forward
    combined_probs, alpha = model(seq, seq_len, bigrec_emb)
    assert combined_probs.shape == (batch_size, num_items + 1)
    assert alpha.shape == (batch_size, 1)
    # Check probs sum to 1
    assert torch.allclose(combined_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    # Test Training Step
    batch = {
        "seq": seq,
        "len_seq": seq_len,
        "next_item": target,
        "bigrec_emb": bigrec_emb
    }
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Test Validation Step
    model.validation_step(batch, 0)
    
    # Test Test Step
    model.test_step(batch, 0)

