import torch
from typing import Dict, Any

class DataBridge:
    """
    教師モデルと生徒モデルの出力を蒸留損失計算に適した形式に橋渡しするクラス。
    DLLM2Recのロジックでは、教師モデルはLLMベース、生徒モデルはIDベースであるため、
    出力形式の整合性を図る役割を担います。
    """
    def __init__(self, num_items: int):
        self.num_items = num_items

    def process_teacher_outputs(self, teacher_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        教師モデルの出力を蒸留に適した形式に処理します。
        現時点では、iLoRAModelの出力が直接利用可能であるため、そのまま返します。

        Args:
            teacher_outputs (Dict[str, Any]): 教師モデルの出力。
                                               例: {"ranking_scores": ..., "embeddings": ...}

        Returns:
            Dict[str, torch.Tensor]: 処理された教師モデルの出力。
                                     例: {"teacher_ranking_scores": ..., "teacher_embeddings": ...}
        """
        # 必要に応じて、ここでLLMの語彙からアイテムIDへのマッピングや、
        # スコアのフィルタリングなどを行う
        return {
            "teacher_ranking_scores": teacher_outputs["ranking_scores"],
            "teacher_embeddings": teacher_outputs["embeddings"]
        }

    def process_student_outputs(self, student_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        生徒モデルの出力を蒸留に適した形式に処理します。
        現時点では、SASRecモデルの出力が直接利用可能であるため、そのまま返します。

        Args:
            student_outputs (Dict[str, Any]): 生徒モデルの出力。
                                               例: {"logits": ..., "embeddings": ...}

        Returns:
            Dict[str, torch.Tensor]: 処理された生徒モデルの出力。
                                     例: {"student_logits": ..., "student_embeddings": ...}
        """
        # 必要に応じて、ここで生徒モデルの出力（例：埋め込み）を抽出・整形する
        return {
            "student_logits": student_outputs["logits"],
            "student_embeddings": student_outputs["embeddings"]
        }

if __name__ == "__main__":
    # テスト用のダミーデータ
    batch_size = 4
    num_items = 100
    teacher_embedding_dim = 128
    student_embedding_dim = 64

    # ダミーの教師モデル出力
    dummy_teacher_outputs = {
        "ranking_scores": torch.randn(batch_size, num_items + 1),
        "embeddings": torch.randn(batch_size, teacher_embedding_dim)
    }

    # ダミーの生徒モデル出力
    dummy_student_outputs = {
        "logits": torch.randn(batch_size, num_items + 1),
        "embeddings": torch.randn(batch_size, student_embedding_dim)
    }

    # DataBridgeのインスタンス化
    data_bridge = DataBridge(num_items=num_items)

    # 教師モデル出力の処理
    processed_teacher = data_bridge.process_teacher_outputs(dummy_teacher_outputs)
    print("--- Processed Teacher Outputs ---")
    print(f"Ranking Scores Shape: {processed_teacher['teacher_ranking_scores'].shape}")
    print(f"Embeddings Shape: {processed_teacher['teacher_embeddings'].shape}")
    assert processed_teacher['teacher_ranking_scores'].shape == (batch_size, num_items + 1)
    assert processed_teacher['teacher_embeddings'].shape == (batch_size, teacher_embedding_dim)

    # 生徒モデル出力の処理
    processed_student = data_bridge.process_student_outputs(dummy_student_outputs)
    print("\n--- Processed Student Outputs ---")
    print(f"Logits Shape: {processed_student['student_logits'].shape}")
    print(f"Embeddings Shape: {processed_student['student_embeddings'].shape}")
    assert processed_student['student_logits'].shape == (batch_size, num_items + 1)
    assert processed_student['student_embeddings'].shape == (batch_size, student_embedding_dim)

    print("\nDataBridge test passed!")
