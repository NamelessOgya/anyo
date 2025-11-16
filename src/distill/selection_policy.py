from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class SelectionPolicy(ABC):
    """
    蒸留にどのサンプルを使用するかを決定するポリシーの抽象基底クラス。
    """
    @abstractmethod
    def select_samples(self, 
                       teacher_outputs: Dict[str, Any], 
                       student_outputs: Dict[str, Any], 
                       batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        与えられた教師モデルと生徒モデルの出力、およびバッチデータに基づいて、
        蒸留に使用するサンプルを選択します。

        Args:
            teacher_outputs (Dict[str, Any]): 教師モデルの出力。
                                               例: {"ranking_scores": ..., "embeddings": ...}
            student_outputs (Dict[str, Any]): 生徒モデルの出力。
                                               例: {"logits": ..., "embeddings": ...}
            batch (Dict[str, torch.Tensor]): 元のバッチデータ。
                                             例: {"item_seq": ..., "next_item": ...}

        Returns:
            Dict[str, Any]: 選択されたサンプルに対応するデータを含む辞書。
                            例: {"selected_teacher_ranking_scores": ..., "selected_student_logits": ...}
                            選択されたサンプルのインデックスやマスクを返すことも可能。
        """
        pass

class AllSamplesPolicy(SelectionPolicy):
    """
    すべてのサンプルを蒸留に使用するポリシー。
    初期実装として、何も選択せずに入力をそのまま返します。
    """
    def select_samples(self, 
                       teacher_outputs: Dict[str, Any], 
                       student_outputs: Dict[str, Any], 
                       batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        すべてのサンプルをそのまま返します。

        Args:
            teacher_outputs (Dict[str, Any]): 教師モデルの出力。
            student_outputs (Dict[str, Any]): 生徒モデルの出力。
            batch (Dict[str, torch.Tensor]): 元のバッチデータ。

        Returns:
            Dict[str, Any]: 入力をそのまま含む辞書。
        """
        return {
            "teacher_ranking_scores": teacher_outputs.get("ranking_scores"),
            "teacher_embeddings": teacher_outputs.get("embeddings"),
            "student_logits": student_outputs.get("logits"),
            "student_embeddings": student_outputs.get("embeddings"),
            "next_item": batch.get("next_item")
        }

if __name__ == "__main__":
    # テスト用のダミーデータ
    batch_size = 4
    num_items = 100
    embedding_dim = 64

    dummy_teacher_outputs = {
        "ranking_scores": torch.randn(batch_size, num_items),
        "embeddings": torch.randn(batch_size, embedding_dim)
    }
    dummy_student_outputs = {
        "logits": torch.randn(batch_size, num_items),
        "embeddings": torch.randn(batch_size, embedding_dim)
    }
    dummy_batch = {
        "item_seq": torch.randint(1, 100, (batch_size, 10)),
        "item_seq_len": torch.randint(1, 10, (batch_size,)),
        "next_item": torch.randint(1, 100, (batch_size,))
    }

    # AllSamplesPolicyのテスト
    all_samples_policy = AllSamplesPolicy()
    selected_data = all_samples_policy.select_samples(
        dummy_teacher_outputs, dummy_student_outputs, dummy_batch
    )

    print("--- AllSamplesPolicy Test ---")
    print(f"Selected data keys: {selected_data.keys()}")
    assert selected_data["teacher_ranking_scores"].shape == (batch_size, num_items)
    assert selected_data["teacher_embeddings"].shape == (batch_size, embedding_dim)
    assert selected_data["student_logits"].shape == (batch_size, num_items)
    assert selected_data["student_embeddings"].shape == (batch_size, embedding_dim)
    assert selected_data["next_item"].shape == (batch_size,)

    print("\nSelectionPolicy and AllSamplesPolicy test passed!")
