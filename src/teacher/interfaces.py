from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, List

class TeacherModel(ABC):
    """
    教師モデルが満たすべきインターフェースを定義します。
    """

    @abstractmethod
    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        入力シーケンスからアイテムの表現やスコアを計算します。
        具体的な出力形式は実装に依存します。

        Args:
            item_seq (torch.Tensor): ユーザーのアイテムシーケンス (batch_size, max_seq_len)。
            item_seq_len (torch.Tensor): 各シーケンスの実際の長さ (batch_size)。

        Returns:
            torch.Tensor: モデルの出力。例えば、各アイテムに対するスコアや、シーケンスの最終表現など。
        """
        pass

    @abstractmethod
    def get_teacher_outputs(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> Dict[str, Any]:
        """
        蒸留に必要な教師モデルの出力を生成します。
        DLLM2Recのロジックを再現するため、ランキング、スコア、埋め込みなどを返します。

        Args:
            item_seq (torch.Tensor): ユーザーのアイテムシーケンス (batch_size, max_seq_len)。
            item_seq_len (torch.Tensor): 各シーケンスの実際の長さ (batch_size)。

        Returns:
            Dict[str, Any]: 蒸留に必要な教師モデルの出力を含む辞書。
                            例: {"ranking_scores": ..., "embeddings": ...}
        """
        pass

if __name__ == "__main__":
    # インターフェースのテスト (抽象クラスなので直接インスタンス化はできない)
    # 継承クラスを作成してテストする
    class DummyTeacher(TeacherModel):
        def __init__(self, num_items: int):
            self.num_items = num_items

        def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
            batch_size = item_seq.shape[0]
            # ダミーのスコアを返す
            return torch.randn(batch_size, self.num_items)

        def get_teacher_outputs(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> Dict[str, Any]:
            batch_size = item_seq.shape[0]
            # ダミーのランキングスコアと埋め込みを返す
            ranking_scores = torch.randn(batch_size, self.num_items)
            embeddings = torch.randn(batch_size, 128) # ダミーの埋め込みサイズ
            return {"ranking_scores": ranking_scores, "embeddings": embeddings}

    dummy_teacher = DummyTeacher(num_items=100)
    
    # ダミーデータ
    item_seq_dummy = torch.randint(1, 100, (4, 10))
    item_seq_len_dummy = torch.randint(1, 10, (4,))

    # forwardテスト
    output = dummy_teacher.forward(item_seq_dummy, item_seq_len_dummy)
    print(f"DummyTeacher forward output shape: {output.shape}")
    assert output.shape == (4, 100)

    # get_teacher_outputsテスト
    teacher_outputs = dummy_teacher.get_teacher_outputs(item_seq_dummy, item_seq_len_dummy)
    print(f"DummyTeacher get_teacher_outputs keys: {teacher_outputs.keys()}")
    print(f"DummyTeacher ranking_scores shape: {teacher_outputs['ranking_scores'].shape}")
    print(f"DummyTeacher embeddings shape: {teacher_outputs['embeddings'].shape}")
    assert teacher_outputs['ranking_scores'].shape == (4, 100)
    assert teacher_outputs['embeddings'].shape == (4, 128)

    print("\nTeacherModel interface test passed!")
