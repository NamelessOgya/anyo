import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RankingDistillationLoss(nn.Module):
    """
    ランキング蒸留損失を計算します。
    教師モデルのランキングスコアと生徒モデルのランキングスコアの差を最小化します。
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): 生徒モデルの出力ロジット (batch_size, num_items)。
            teacher_logits (torch.Tensor): 教師モデルの出力ロジット (batch_size, num_items)。

        Returns:
            torch.Tensor: ランキング蒸留損失。
        """
        # ソフトターゲットの計算
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KLダイバージェンス損失
        loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2) # 温度スケーリングの補正

        return loss

class EmbeddingDistillationLoss(nn.Module):
    """
    埋め込み蒸留損失を計算します。
    教師モデルの埋め込みと生徒モデルの埋め込みの間の距離を最小化します。
    """
    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unsupported embedding distillation loss type: {loss_type}")
        self.loss_type = loss_type

    def forward(self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_embeddings (torch.Tensor): 生徒モデルの埋め込み (batch_size, embedding_dim)。
            teacher_embeddings (torch.Tensor): 教師モデルの埋め込み (batch_size, embedding_dim)。

        Returns:
            torch.Tensor: 埋め込み蒸留損失。
        """
        if self.loss_type == "cosine":
            # CosineEmbeddingLossはターゲットラベルを必要とする (1: 類似, -1: 非類似)
            # ここでは類似度を最大化したいので、ターゲットを1とする
            target = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
            loss = self.loss_fn(student_embeddings, teacher_embeddings, target)
        else: # mse
            loss = self.loss_fn(student_embeddings, teacher_embeddings)
        return loss

if __name__ == "__main__":
    # テスト用のダミーデータ
    batch_size = 4
    num_items = 100
    embedding_dim = 64

    student_logits = torch.randn(batch_size, num_items, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_items)

    student_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    teacher_embeddings = torch.randn(batch_size, embedding_dim)

    # ランキング蒸留損失のテスト
    ranking_loss_fn = RankingDistillationLoss(temperature=2.0)
    ranking_loss = ranking_loss_fn(student_logits, teacher_logits)
    print(f"Ranking Distillation Loss: {ranking_loss.item()}")
    ranking_loss.backward() # 勾配計算の確認
    print(f"Student logits grad (ranking): {student_logits.grad.norm().item()}")

    # 埋め込み蒸留損失 (MSE) のテスト
    embedding_mse_loss_fn = EmbeddingDistillationLoss(loss_type="mse")
    embedding_mse_loss = embedding_mse_loss_fn(student_embeddings, teacher_embeddings)
    print(f"Embedding Distillation Loss (MSE): {embedding_mse_loss.item()}")
    student_embeddings.grad = None # 勾配をリセット
    embedding_mse_loss.backward()
    print(f"Student embeddings grad (MSE): {student_embeddings.grad.norm().item()}")

    # 埋め込み蒸留損失 (Cosine) のテスト
    embedding_cosine_loss_fn = EmbeddingDistillationLoss(loss_type="cosine")
    embedding_cosine_loss = embedding_cosine_loss_fn(student_embeddings, teacher_embeddings)
    print(f"Embedding Distillation Loss (Cosine): {embedding_cosine_loss.item()}")
    student_embeddings.grad = None # 勾配をリセット
    embedding_cosine_loss.backward()
    print(f"Student embeddings grad (Cosine): {student_embeddings.grad.norm().item()}")

    print("\nKnowledge Distillation Losses test passed!")
