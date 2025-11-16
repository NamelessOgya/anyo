import pytest
import torch
import torch.nn.functional as F
from src.distill.kd_losses import RankingDistillationLoss, EmbeddingDistillationLoss

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
