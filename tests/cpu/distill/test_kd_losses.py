import pytest
import torch
import torch.nn.functional as F
from src.distill.kd_losses import RankingDistillationLoss, EmbeddingDistillationLoss, PropensityScoreCalculator, DROLoss, WeightedBCELoss
from collections import Counter

def test_ranking_distillation_loss():
    """
    RankingDistillationLossのテスト。
    """
    batch_size = 4
    num_items = 100
    
    student_logits = torch.randn(batch_size, num_items)
    teacher_logits = torch.randn(batch_size, num_items)
    
    # 1. 教師と生徒のロジットが同じ場合、損失が0に近くなるか
    loss_fn_same = RankingDistillationLoss(temperature=1.0)
    loss_same = loss_fn_same(student_logits, student_logits)
    assert torch.allclose(loss_same, torch.tensor(0.0), atol=1e-6)
    
    # 2. 温度Tを大きくすると、損失が小さくなるか
    loss_fn_t1 = RankingDistillationLoss(temperature=1.0)
    loss_t1 = loss_fn_t1(student_logits, teacher_logits)
    
    loss_fn_t10 = RankingDistillationLoss(temperature=10.0)
    loss_t10 = loss_fn_t10(student_logits, teacher_logits)
    
    # 温度を上げると、確率分布がより滑らかになり、KLダイバージェンスは小さくなる傾向がある
    # ただし、損失には T^2 が乗算されるため、一概には言えない
    # ここでは、KL(p_student || p_teacher) の計算が正しく行われるかを確認する
    
    # 手動で計算
    temp = 1.0
    manual_loss = F.kl_div(
        F.log_softmax(student_logits / temp, dim=-1),
        F.softmax(teacher_logits / temp, dim=-1),
        reduction='batchmean'
    ) * (temp ** 2)
    
    assert torch.allclose(loss_t1, manual_loss)

def test_embedding_distillation_loss_mse():
    """
    EmbeddingDistillationLoss (MSE) のテスト。
    """
    batch_size = 4
    embedding_dim = 64
    
    student_embeddings = torch.randn(batch_size, embedding_dim)
    
    # 1. 教師と生徒の埋め込みが同じ場合、損失が0になるか
    loss_fn = EmbeddingDistillationLoss(loss_type="mse")
    loss_same = loss_fn(student_embeddings, student_embeddings)
    assert torch.allclose(loss_same, torch.tensor(0.0))
    
    # 2. 損失が正しく計算されるか
    teacher_embeddings = torch.randn(batch_size, embedding_dim)
    loss = loss_fn(student_embeddings, teacher_embeddings)
    manual_loss = F.mse_loss(student_embeddings, teacher_embeddings)
    assert torch.allclose(loss, manual_loss)

def test_embedding_distillation_loss_cosine():
    """
    EmbeddingDistillationLoss (Cosine) のテスト。
    """
    batch_size = 4
    embedding_dim = 64
    
    student_embeddings = torch.randn(batch_size, embedding_dim)
    
    # 1. 教師と生徒の埋め込みが同じ場合、損失が0になるか
    loss_fn = EmbeddingDistillationLoss(loss_type="cosine")
    loss_same = loss_fn(student_embeddings, student_embeddings)
    assert torch.allclose(loss_same, torch.tensor(0.0))
    
    # 2. 損失が正しく計算されるか
    teacher_embeddings = torch.randn(batch_size, embedding_dim)
    loss = loss_fn(student_embeddings, teacher_embeddings)
    target = torch.ones(batch_size)
    manual_loss = F.cosine_embedding_loss(student_embeddings, teacher_embeddings, target)
    assert torch.allclose(loss, manual_loss)

def test_propensity_score_calculator():
    """
    PropensityScoreCalculatorのテスト。
    """
    item_num = 10
    train_next_items = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    power = 0.5
    
    calculator = PropensityScoreCalculator(item_num, train_next_items, power)
    ps = calculator.get_ps()
    
    assert ps.shape == (item_num,)
    # 特定のアイテムのpsが期待通りか (手計算で確認)
    # freq = {1:1, 2:2, 3:3, 4:4}
    # pop = [0, 1, 2, 3, 4, 0, 0, 0, 0, 0] (item 0, 5-9 have 0 freq)
    # ps_raw = [1, 2, 3, 4, 5, 1, 1, 1, 1, 1]
    # ps_norm = ps_raw / sum(ps_raw) = ps_raw / 20
    # ps_powered = (ps_norm)^0.5
    expected_ps_raw = torch.tensor([1, 2, 3, 4, 5, 1, 1, 1, 1, 1], dtype=torch.float)
    expected_ps_norm = expected_ps_raw / expected_ps_raw.sum()
    expected_ps = torch.pow(expected_ps_norm, power)
    assert torch.allclose(ps, expected_ps, atol=1e-6)

def test_dro_loss():
    """
    DROLossのテスト。
    """
    batch_size = 4
    num_items = 10
    # psをnum_items + 1のサイズで作成
    ps = torch.rand(num_items + 1)
    ps = ps / ps.sum() # 正規化
    beta = 1.0
    
    model_output = torch.randn(batch_size, num_items, requires_grad=True)
    target = torch.randint(0, num_items, (batch_size,))
    
    dro_loss_fn = DROLoss(ps=ps, beta=beta)
    loss = dro_loss_fn(model_output, target)
    print(f"DRO Loss: {loss.item()}")
    model_output.grad = None # 勾配をリセット
    loss.backward()
    print(f"Model output grad (DRO): {model_output.grad.norm().item()}")

def test_weighted_bce_loss_no_dro():
    """
    WeightedBCELoss (DROなし) のテスト。
    """
    batch_size = 4
    num_items = 10
    num_candidates = 3
    
    student_logits = torch.randn(batch_size, num_items, requires_grad=True)
    teacher_candidates = torch.randint(0, num_items, (batch_size, num_candidates))
    weights = torch.rand(batch_size, num_candidates)
    weights = weights / weights.sum(dim=1, keepdim=True) # 正規化
    
    loss_fn = WeightedBCELoss(alpha=0.0)
    neg_samples = torch.randint(0, num_items, (batch_size, 5)) # Add dummy neg_samples
    loss = loss_fn(student_logits, teacher_candidates, weights, neg_samples)
    
    assert loss.shape == torch.Size([])
    assert loss.requires_grad
    
    loss.backward()
    assert student_logits.grad is not None

def test_weighted_bce_loss_with_dro():
    """
    WeightedBCELoss (DROあり) のテスト。
    """
    batch_size = 4
    num_items = 10
    num_candidates = 3
    
    student_logits = torch.randn(batch_size, num_items, requires_grad=True)
    teacher_candidates = torch.randint(0, num_items, (batch_size, num_candidates))
    weights = torch.rand(batch_size, num_candidates)
    weights = weights / weights.sum(dim=1, keepdim=True) # 正規化

    # alpha=0 (no DRO)
    weighted_bce_loss_no_dro_fn = WeightedBCELoss(alpha=0.0)
    neg_samples = torch.randint(0, num_items, (batch_size, 5)) # Add dummy neg_samples
    loss_no_dro = weighted_bce_loss_no_dro_fn(student_logits, teacher_candidates, weights, neg_samples)
    print(f"WeightedBCELoss (no DRO): {loss_no_dro.item()}")
    student_logits.grad = None
    loss_no_dro.backward()
    print(f"Student logits grad (WeightedBCELoss no DRO): {student_logits.grad.norm().item()}")

    # alpha > 0 (with DRO)
    # psをnum_items + 1のサイズで作成
    ps_for_weighted_bce = torch.rand(num_items + 1)
    ps_for_weighted_bce = ps_for_weighted_bce / ps_for_weighted_bce.sum() # 正規化
    weighted_bce_loss_with_dro_fn = WeightedBCELoss(alpha=0.5, ps=ps_for_weighted_bce, beta=1.0)
    loss_with_dro = weighted_bce_loss_with_dro_fn(student_logits, teacher_candidates, weights, neg_samples)
    print(f"WeightedBCELoss (with DRO): {loss_with_dro.item()}")
    student_logits.grad = None
    loss_with_dro.backward()
    print(f"Student logits grad (WeightedBCELoss with DRO): {student_logits.grad.norm().item()}")


def test_weighted_bce_loss_dro_alpha_zero_no_ps_error():
    """
    WeightedBCELossでalpha=0の場合、psがNoneでもエラーにならないことのテスト。
    """
    loss_fn = WeightedBCELoss(alpha=0.0, ps=None, beta=1.0)
    assert loss_fn.alpha == 0.0
    assert loss_fn.ps is None

def test_weighted_bce_loss_dro_alpha_nonzero_no_ps_error():
    """
    WeightedBCELossでalpha>0の場合、psがNoneだとValueErrorが発生することのテスト。
    """
    with pytest.raises(ValueError, match="Propensity scores \(ps\) must be provided if alpha > 0 for DROLoss."):
        WeightedBCELoss(alpha=0.1, ps=None, beta=1.0)
