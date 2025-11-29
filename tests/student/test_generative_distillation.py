import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from unittest.mock import MagicMock
import pytest
import sys
import os

from src.student.models import SASRec
from src.teacher.generative_ranker import GenerativeRanker
from src.student.generative_distillation_trainer import GenerativeDistillationTrainer

class DummyDataset(Dataset):
    def __init__(self, num_samples=10, seq_len=50, num_items=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_items = num_items

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "item_seq": torch.randint(1, self.num_items, (self.seq_len,)),
            "item_seq_len": torch.tensor(self.seq_len),
            "next_item": torch.randint(1, self.num_items, (1,)), # 1-based ID
        }

@pytest.fixture
def device():
    return "cpu"

@pytest.fixture
def student_model(device):
    return SASRec(
        num_items=100,
        hidden_size=32,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=50,
        teacher_embedding_dim=64 # Teacher hidden size
    ).to(device)

@pytest.fixture
def teacher_model(device):
    teacher = MagicMock(spec=GenerativeRanker)
    teacher.device = device
    teacher.eval = MagicMock()
    teacher.create_prompt = MagicMock(return_value="Dummy Prompt")
    
    # Mock generate_and_extract_state
    def mock_generate(*args, **kwargs):
        # args[0] is prompts list
        batch_size = len(args[0]) if args else 4
        texts = ["[0] > [1] > [2]"] * batch_size
        states = torch.randn(batch_size, 64).to(device)
        return texts, states
    
    teacher.generate_and_extract_state = MagicMock(side_effect=mock_generate)
    teacher.parse_ranking = MagicMock(return_value=[0, 1, 2]) # Dummy ranking
    return teacher

def test_generative_distillation_training_step(student_model, teacher_model, device):
    item_id_to_name = {i: f"Item_{i}" for i in range(101)}
    
    trainer = GenerativeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        learning_rate=0.001,
        lambda_emb=1.0,
        lambda_rank=1.0,
        num_candidates=20,
        item_id_to_name=item_id_to_name
    )
    
    dataloader = DataLoader(DummyDataset(), batch_size=4)
    batch = next(iter(dataloader))
    
    # Call training_step
    loss = trainer.training_step(batch, 0)
    
    assert loss.item() > 0
    print(f"Total Loss: {loss.item()}")
    
    # Verify backward pass
    optimizer = trainer.configure_optimizers()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check if gradients are computed for key components
    assert student_model.item_embeddings.weight.grad is not None
    assert student_model.position_embeddings.weight.grad is not None
    
    # Optional: Check transformer weights
    for param in student_model.transformer_blocks.parameters():
        if param.requires_grad:
             assert param.grad is not None

def test_rank_to_score(student_model, teacher_model):
    trainer = GenerativeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model
    )
    
    rankings = [[0, 1, 2], [2, 0, 1]]
    num_candidates = 3
    scores = trainer.rank_to_score(rankings, num_candidates, "cpu")
    
    assert scores.shape == (2, 3)
    # Rank 0 (1st) -> 1/log2(2) = 1.0
    # Rank 1 (2nd) -> 1/log2(3) approx 0.63
    # Rank 2 (3rd) -> 1/log2(4) = 0.5
    
    assert torch.isclose(scores[0, 0], torch.tensor(1.0))
    assert torch.isclose(scores[0, 1], torch.tensor(1.0/torch.log2(torch.tensor(3.0))))
    assert torch.isclose(scores[0, 2], torch.tensor(0.5))

from unittest.mock import MagicMock, patch

def test_generative_ranker_logic():
    # Test logic that doesn't require loading the LLM
    tokenizer = MagicMock()
    
    with patch("src.teacher.generative_ranker.AutoModelForCausalLM") as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = MagicMock()
        
        ranker = GenerativeRanker(
            model_name_or_path="dummy",
            tokenizer=tokenizer,
            device="cpu"
        )
    
    # 1. Test create_prompt
    history = "ItemA | ItemB"
    candidates = ["Cand1", "Cand2"]
    prompt = ranker.create_prompt(history, candidates)
    
    assert "ItemA | ItemB" in prompt
    assert "[1] Cand1" in prompt
    assert "[2] Cand2" in prompt
    assert "Rank the 2 passages" in prompt
    
    # 2. Test parse_ranking
    # Case A: Standard output
    generated_text = "The ranking is [2] > [1]"
    ranking = ranker.parse_ranking(generated_text, num_candidates=2)
    assert ranking == [1, 0] # [2] is index 1, [1] is index 0
    
    # Case B: Partial output (missing item appended)
    generated_text = "[2] is the best"
    ranking = ranker.parse_ranking(generated_text, num_candidates=2)
    assert ranking == [1, 0] # 1 is found, 0 is missing so appended
    
    # Case C: Out of bounds or garbage
    generated_text = "[99] > [1]"
    ranking = ranker.parse_ranking(generated_text, num_candidates=2)
    assert ranking == [0, 1] # [99] ignored. [1] (idx 0) found? No, [1] is idx 0.
    # Wait, regex finds "1". idx = 0.
    # If text is "[99] > [1]", matches are "99", "1".
    # idx 98 (invalid), idx 0 (valid).
    # So ranking should have [0]. Then missing [1] appended. -> [0, 1].
    
    # Let's verify Case C logic carefully.
    # matches: ['99', '1']
    # idx: 98 (ignored), 0 (added)
    # ranking: [0]
    # missing: 1
    # result: [0, 1]
    assert ranking == [0, 1]

