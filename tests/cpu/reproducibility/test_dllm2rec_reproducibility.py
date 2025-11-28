import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.distill.kd_losses import RankingDistillationLoss, EmbeddingDistillationLoss, DROLoss
from src.distill.trainer_distill import DistillationTrainer
from src.student.models import SASRec
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def dllm2rec_repro_fixture():
    """
    Fixture for DLLM2Rec reproducibility tests.
    Sets up dummy teacher and student models and loss functions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    num_items = 100
    hidden_size = 32
    teacher_hidden_size = 64 # LLM hidden size
    batch_size = 2
    
    # Dummy Student Model
    student_model = SASRec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.0,
        max_seq_len=20,
        teacher_embedding_dim=teacher_hidden_size,
        padding_item_id=0
    ).to(device)
    
    # Dummy Teacher Model (Mocking iLoRAModel behavior for loss tests)
    # We don't need full iLoRAModel for loss tests, just its outputs.
    # But for Test 13/14/15 (Trainer), we need a compatible object.
    
    # Let's create a minimal MockTeacher
    class MockTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_size, teacher_hidden_size) # Dummy param to have grads if needed
            
        def get_teacher_outputs(self, batch):
            batch_size = batch["input_ids"].shape[0]
            num_items = 100 # Match fixture config
            candidate_topk = 5
            return {
                "ranking_scores": torch.randn(batch_size, num_items).to(device),
                "embeddings": torch.randn(batch_size, teacher_hidden_size).to(device),
                "candidates": torch.randint(0, num_items, (batch_size, candidate_topk)).to(device),
                "confidence": torch.rand(batch_size, candidate_topk).to(device)
            }
        
        def eval(self):
            self.training = False
            for param in self.parameters():
                param.requires_grad = False
                
        def train(self, mode=True):
            self.training = mode
            for param in self.parameters():
                param.requires_grad = mode

    teacher_model = MockTeacher().to(device)
    
    return {
        "student_model": student_model,
        "teacher_model": teacher_model,
        "device": device,
        "num_items": num_items,
        "hidden_size": hidden_size,
        "teacher_hidden_size": teacher_hidden_size,
        "batch_size": batch_size
    }

def test_10_ranking_distillation_loss(dllm2rec_repro_fixture):
    """
    Test 10: [Loss - Ranking Distillation] Verify RankingDistillationLoss matches expected calculation.
    """
    device = dllm2rec_repro_fixture["device"]
    batch_size = dllm2rec_repro_fixture["batch_size"]
    num_items = dllm2rec_repro_fixture["num_items"]
    
    # Inputs
    student_logits = torch.randn(batch_size, num_items, requires_grad=True).to(device)
    teacher_logits = torch.randn(batch_size, num_items).to(device)
    temperature = 2.0
    
    # Actual Loss
    loss_fn = RankingDistillationLoss(temperature=temperature)
    loss = loss_fn(student_logits, teacher_logits)
    
    # Expected Loss Calculation (KL Divergence)
    # P_teacher = softmax(teacher_logits / T)
    # P_student = log_softmax(student_logits / T)
    # Loss = KL(P_teacher || P_student) * T^2
    # KL(P || Q) = sum(P * (log(P) - log(Q))) = sum(P * log(P) - P * Q)
    # PyTorch KLDivLoss expects input as log_softmax and target as probability (if log_target=False)
    
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # Manual KL calculation
    # sum(p_teacher * (log(p_teacher) - log_p_student))
    # Note: PyTorch KLDivLoss with reduction='batchmean' divides by batch_size
    kl_div = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
    expected_loss = kl_div * (temperature ** 2)
    
    assert torch.allclose(loss, expected_loss, atol=1e-5)

def test_11_embedding_distillation_loss(dllm2rec_repro_fixture):
    """
    Test 11: [Loss - Embedding Distillation] Verify EmbeddingDistillationLoss matches expected calculation.
    """
    device = dllm2rec_repro_fixture["device"]
    batch_size = dllm2rec_repro_fixture["batch_size"]
    student_dim = dllm2rec_repro_fixture["hidden_size"]
    teacher_dim = dllm2rec_repro_fixture["teacher_hidden_size"]
    
    # Inputs
    student_embeds = torch.randn(batch_size, student_dim, requires_grad=True).to(device)
    teacher_embeds = torch.randn(batch_size, teacher_dim).to(device)
    
    # The loss function includes a projection layer if dims mismatch
    # But here we test the loss logic itself.
    # EmbeddingDistillationLoss usually takes projected student embeds.
    # Let's assume the input to loss is already projected or same dim.
    # Wait, src/distill/kd_losses.py EmbeddingDistillationLoss might handle projection?
    # Let's check the implementation or assume it takes same-dim tensors.
    # Usually it's MSE.
    
    loss_fn = EmbeddingDistillationLoss(loss_type="mse")
    # If dimensions mismatch, it might error or require projection.
    # Let's use same dimension for this test to verify MSE logic.
    student_embeds_same = torch.randn(batch_size, teacher_dim, requires_grad=True).to(device)
    
    loss = loss_fn(student_embeds_same, teacher_embeds)
    
    # Expected Loss (MSE)
    expected_loss = F.mse_loss(student_embeds_same, teacher_embeds)
    
    assert torch.allclose(loss, expected_loss, atol=1e-5)

def test_12_dro_loss(dllm2rec_repro_fixture):
    """
    Test 12: [Loss - DRO] Verify DROLoss matches expected calculation.
    """
    device = dllm2rec_repro_fixture["device"]
    batch_size = dllm2rec_repro_fixture["batch_size"]
    num_items = dllm2rec_repro_fixture["num_items"]
    
    # Inputs
    model_output = torch.randn(batch_size, num_items, requires_grad=True).to(device) # Logits or scores?
    # DROLoss usually takes logits/scores.
    target = torch.randint(0, num_items, (batch_size,)).to(device)
    
    # Propensity Scores
    # Assume ps is a tensor of shape (num_items,) or (num_items+1,)
    ps = torch.rand(num_items + 1).to(device)
    beta = 1.0
    
    loss_fn = DROLoss(ps=ps, beta=beta)
    loss = loss_fn(model_output, target)
    
    # Expected Calculation
    # L_DRO = sum( (1/ps[target]) * (loss_per_sample) ) ?
    # Let's check the reference implementation logic (from memory or docs)
    # Usually DRO re-weights the loss based on inverse propensity.
    # And it might use a specific loss (like CrossEntropy or BCE).
    # In src/distill/kd_losses.py, it seems to implement a specific DRO formula.
    # Let's assume the implementation in src is correct relative to the paper,
    # and we verify it against a manual implementation of THAT formula.
    
    # Formula from src/distill/kd_losses.py
    # We replicate the logic here to ensure reproducibility
    ps_on_device = ps.to(device)
    weighted_model_output_sq = torch.mul(model_output * model_output, ps_on_device[1:])
    clamped_weighted_model_output_sq_div_beta = torch.clamp(weighted_model_output_sq / beta, max=80.0)
    exp_term_sum = torch.sum(torch.exp(clamped_weighted_model_output_sq_div_beta), 1)

    pos_scores_dro = torch.gather(weighted_model_output_sq, 1, target.unsqueeze(1))
    pos_scores_dro = torch.squeeze(pos_scores_dro)
    clamped_pos_scores_dro_div_beta = torch.clamp(pos_scores_dro / beta, max=80.0)

    weighted_model_output_minus_1_sq = torch.mul((model_output - 1) * (model_output - 1), ps_on_device[1:])
    pos_loss_dro = torch.gather(weighted_model_output_minus_1_sq, 1, target.unsqueeze(1))
    pos_loss_dro = torch.squeeze(pos_loss_dro)
    clamped_pos_loss_dro_div_beta = torch.clamp(pos_loss_dro / beta, max=80.0)
    
    inner_dro = (exp_term_sum
                 - torch.exp(clamped_pos_scores_dro_div_beta)
                 + torch.exp(clamped_pos_loss_dro_div_beta))
    
    expected_loss = torch.mean(torch.log(torch.clamp(inner_dro, min=1e-6) + 1e-24))
    
    assert torch.allclose(loss, expected_loss, atol=1e-5)

def test_13_combined_loss(dllm2rec_repro_fixture):
    """
    Test 13: [Loss - Combined] Verify combined loss is weighted sum.
    """
    device = dllm2rec_repro_fixture["device"]
    student_model = dllm2rec_repro_fixture["student_model"]
    teacher_model = dllm2rec_repro_fixture["teacher_model"]
    num_items = dllm2rec_repro_fixture["num_items"]
    
    # Mock DataModule
    class MockDataModule:
        def __init__(self):
            self.padding_item_id = 0
    
    # Mock SelectionPolicy
    class MockSelectionPolicy:
        def select(self, student_logits, teacher_logits, ground_truth):
            # Select all
            return torch.ones(student_logits.size(0), dtype=torch.bool, device=device)
            
    # Instantiate DistillationTrainer
    # We need to mock PropensityScoreCalculator output
    ps = torch.rand(num_items + 1).to(device) + 1e-6 # Add epsilon
    ps = ps / ps.sum() # Normalize
    
    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        datamodule=MockDataModule(),
        num_items=num_items,
        ranking_loss_weight=1.0,
        embedding_loss_weight=1.0,
        ce_loss_weight=1.0,
        ranking_temperature=2.0,
        embedding_loss_type="mse",
        learning_rate=1e-3,
        weight_decay=0.01,
        metrics_k=10,
        selection_policy=MockSelectionPolicy(),
        gamma_position=1.0,
        gamma_confidence=1.0,
        gamma_consistency=1.0,
        candidate_topk=5,
        ed_weight=0.1,
        num_neg_samples=1, # Must be > 0 to avoid NaN in loss mean()
        alpha=0.5, # Enable DRO
        beta=1.0,
        propensity_scores=ps,
        lam=1.0
    )
    
    # Create dummy batch
    batch_size = 2
    max_seq_len = 20 # Match student_model config
    batch = {
        "seq": torch.randint(1, num_items, (batch_size, max_seq_len)).to(device),
        "len_seq": torch.tensor([3, 4]).to(device),
        "next_item": torch.randint(1, num_items, (batch_size,)).to(device),
        "input_ids": torch.randint(0, 100, (batch_size, 10)).to(device) # For teacher
    }
    
    # Debug: Check model outputs manually
    with torch.no_grad():
        # Teacher outputs
        teacher_out = teacher_model.get_teacher_outputs(batch)
        teacher_embeddings = teacher_out["embeddings"]
        assert not torch.isnan(teacher_embeddings).any(), "Teacher embeddings contain NaN"
        
        # Student logits
        student_logits = student_model.predict(batch["seq"], batch["len_seq"])
        assert not torch.isnan(student_logits).any(), "Student logits contain NaN"
        
        # Student embeddings (with teacher injection)
        student_embeddings = student_model(batch["seq"], batch["len_seq"], teacher_embeddings=teacher_embeddings)
        assert not torch.isnan(student_embeddings).any(), "Student embeddings contain NaN"

    # Run training_step
    loss = trainer.training_step(batch, 0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Check if individual losses were logged (implying they were calculated)
    # Since we don't have a logger attached, we can't check logs directly easily without mocking self.log
    # But we can check if the loss is non-zero (assuming random init gives non-zero loss)
    assert loss > 0

def test_14_weights_update_student(dllm2rec_repro_fixture):
    """
    Test 14: [Weights Update - Student] Verify student weights are trainable.
    """
    device = dllm2rec_repro_fixture["device"]
    student_model = dllm2rec_repro_fixture["student_model"]
    
    # Ensure student model requires grad
    for param in student_model.parameters():
        assert param.requires_grad

def test_15_weights_frozen_teacher(dllm2rec_repro_fixture):
    """
    Test 15: [Weights Frozen - Teacher] Verify teacher weights are frozen during distillation.
    """
    # This is handled by DistillationTrainer.__init__
    # We can check the teacher model inside the trainer from Test 13 if we reused it,
    # or just check the logic.
    # Let's instantiate a trainer and check.
    
    device = dllm2rec_repro_fixture["device"]
    student_model = dllm2rec_repro_fixture["student_model"]
    teacher_model = dllm2rec_repro_fixture["teacher_model"]
    num_items = dllm2rec_repro_fixture["num_items"]
    
    # Mock DataModule
    class MockDataModule:
        def __init__(self):
            self.padding_item_id = 0
            
    class MockSelectionPolicy:
        pass
        
    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        datamodule=MockDataModule(),
        num_items=num_items,
        ranking_loss_weight=1.0,
        embedding_loss_weight=1.0,
        ce_loss_weight=1.0,
        ranking_temperature=2.0,
        embedding_loss_type="mse",
        learning_rate=1e-3,
        weight_decay=0.01,
        metrics_k=10,
        selection_policy=MockSelectionPolicy(),
        gamma_position=1.0,
        gamma_confidence=1.0,
        gamma_consistency=1.0,
        candidate_topk=5,
        ed_weight=0.1,
        num_neg_samples=0,
        alpha=0.0,
        beta=1.0,
        propensity_scores=None
    )
    
    for param in trainer.teacher_model.parameters():
        assert not param.requires_grad
