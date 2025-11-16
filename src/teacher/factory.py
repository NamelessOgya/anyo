from omegaconf import DictConfig
from src.teacher.interfaces import TeacherModel
from src.teacher.ilora_model import iLoRAModel
import torch
from typing import Dict

def create_teacher_model(cfg: DictConfig, num_items: int, max_seq_len: int, item_id_to_name: Dict[int, str], padding_item_id: int) -> TeacherModel:
    """
    Hydraの設定に基づいて教師モデルのインスタンスを生成します。

    Args:
        cfg (DictConfig): Hydraの設定オブジェクト。
        num_items (int): アイテムの総数。
        max_seq_len (int): シーケンスの最大長。
        item_id_to_name (Dict[int, str]): アイテムIDから名前へのマッピング。
        padding_item_id (int): パディング用のアイテムID。

    Returns:
        TeacherModel: 構築された教師モデルのインスタンス。
    """
    model_type = cfg.teacher.model_type

    if model_type == "ilora":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = iLoRAModel(
            llm_model_name=cfg.teacher.llm_model_name,
            num_lora_experts=cfg.teacher.num_lora_experts,
            lora_r=cfg.teacher.lora_r,
            lora_alpha=cfg.teacher.lora_alpha,
            lora_dropout=cfg.teacher.lora_dropout,
            num_items=num_items,
            max_seq_len=max_seq_len,
            hidden_size=cfg.teacher.hidden_size,
            dropout_rate=cfg.teacher.dropout_rate,
            item_id_to_name=item_id_to_name,
            padding_item_id=padding_item_id, # 追加
            device=device
        )
        return model
    else:
        raise ValueError(f"Unknown teacher model type: {model_type}")

if __name__ == "__main__":
    # テスト用のダミーHydra設定
    from omegaconf import OmegaConf

    # ダミー設定
    cfg = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 3,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "hidden_size": 64,
            "dropout_rate": 0.1
        }
    })

    num_items_dummy = 5000
    max_seq_len_dummy = 50
    dummy_item_id_to_name = {i: f"Item {i}" for i in range(num_items_dummy + 1)}
    padding_item_id_dummy = 0

    # モデルの生成
    teacher_model = create_teacher_model(
        cfg, 
        num_items_dummy, 
        max_seq_len_dummy, 
        dummy_item_id_to_name,
        padding_item_id_dummy
    )
    print(f"Created teacher model type: {type(teacher_model)}")

    # インターフェースを満たしているか確認
    assert isinstance(teacher_model, TeacherModel)
    assert isinstance(teacher_model, torch.nn.Module)

    # ダミーデータでforwardとget_teacher_outputsをテスト
    item_seq_dummy = torch.randint(1, num_items_dummy, (4, max_seq_len_dummy)).to(teacher_model.device)
    item_seq_len_dummy = torch.randint(1, max_seq_len_dummy + 1, (4,)).to(teacher_model.device)

    output_scores = teacher_model(item_seq_dummy, item_seq_len_dummy)
    print(f"Forward output scores shape: {output_scores.shape}")

    teacher_outputs = teacher_model.get_teacher_outputs(item_seq_dummy, item_seq_len_dummy)
    print(f"Teacher outputs keys: {teacher_outputs.keys()}")

    print("\nTeacher model factory test passed!")
