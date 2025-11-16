from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class SelectionPolicy(ABC):
    """
    蒸留にどのサンプルを使用するかを決定するポリシーの抽象基底クラス。
    """
    @abstractmethod
    def select(self, 
               student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               ground_truth: torch.Tensor) -> torch.Tensor:
        """
        与えられた教師モデルと生徒モデルの出力、および正解データに基づいて、
        蒸留に使用するサンプルを選択するためのブールマスクを返します。

        Args:
            student_logits (torch.Tensor): 生徒モデルの出力ロジット。
            teacher_logits (torch.Tensor): 教師モデルの出力ロジット。
            ground_truth (torch.Tensor): 正解アイテムのインデックス。

        Returns:
            torch.Tensor: 蒸留に使用するサンプルを示すブールマスク (batch_size,)
        """
        pass

class AllSamplesPolicy(SelectionPolicy):
    """
    すべてのサンプルを蒸留に使用するポリシー。
    """
    def select(self, 
               student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               ground_truth: torch.Tensor) -> torch.Tensor:
        """
        すべてのサンプルを選択するためのマスク (すべてTrue) を返します。
        """
        batch_size = student_logits.size(0)
        return torch.ones(batch_size, dtype=torch.bool, device=student_logits.device)

if __name__ == "__main__":
    # テスト用のダミーデータ
    batch_size = 4
    num_items = 100
    embedding_dim = 64

    dummy_student_logits = torch.randn(batch_size, num_items)
    dummy_teacher_logits = torch.randn(batch_size, num_items)
    dummy_ground_truth = torch.randint(1, 100, (batch_size,))

    # AllSamplesPolicyのテスト
    all_samples_policy = AllSamplesPolicy()
    distill_mask = all_samples_policy.select(
        dummy_student_logits, dummy_teacher_logits, dummy_ground_truth
    )

    print("--- AllSamplesPolicy Test ---")
    print(f"Distill mask: {distill_mask}")
    assert distill_mask.shape == (batch_size,)
    assert distill_mask.all()

    print("\nSelectionPolicy and AllSamplesPolicy test passed!")
