from omegaconf import DictConfig
from src.teacher.interfaces import TeacherModel
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector # MLPProjectorをインポート
import torch
import torch.nn as nn # Added import
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer # LLMとTokenizerをロードするために追加
from src.student.models import SASRec # Import SASRec

def create_teacher_model(cfg: DictConfig, num_items: int, max_seq_len: int, item_id_to_name: Dict[int, str], padding_item_id: int, candidate_topk: int) -> TeacherModel:
    """
    Hydraの設定に基づいて教師モデルのインスタンスを生成します。

    Args:
        cfg (DictConfig): Hydraの設定オブジェクト。
        num_items (int): アイテムの総数。
        max_seq_len (int): シーケンスの最大長。
        item_id_to_name (Dict[int, str]): アイテムIDから名前へのマッピング。
        padding_item_id (int): パディング用のアイテムID。
        candidate_topk (int): 候補アイテムのトップK。

    Returns:
        TeacherModel: 構築された教師モデルのインスタンス。
    """
    model_type = cfg.teacher.model_type

    if model_type == "ilora":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # LLMとTokenizerのロード
        llm = AutoModelForCausalLM.from_pretrained(cfg.teacher.llm_model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name)
        if tokenizer.pad_token is None: # Ensure pad_token is set
            tokenizer.pad_token = tokenizer.eos_token
        
        # 特殊トークンを追加
        tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
        llm.resize_token_embeddings(len(tokenizer)) # トークン埋め込み層のサイズを調整

        # rec_modelのロード
        # Check if a pre-trained rec_model checkpoint path is provided
        if cfg.teacher.get("rec_model_checkpoint_path"):
            print(f"Loading pre-trained SASRec model from {cfg.teacher.rec_model_checkpoint_path}")
            # Instantiate SASRec model first to load state_dict into it
            rec_model = SASRec(
                num_items=num_items,
                hidden_size=cfg.student.hidden_size,
                num_heads=cfg.student.num_heads,
                num_layers=cfg.student.num_layers,
                dropout_rate=cfg.student.dropout_rate,
                max_seq_len=max_seq_len,
            ).to(device)
            # Load the state_dict
            rec_model.load_state_dict(torch.load(cfg.teacher.rec_model_checkpoint_path, map_location=device))
            rec_model.eval() # Set to eval mode
        else:
            print("No pre-trained SASRec checkpoint path provided. Initializing a new SASRec model.")
            rec_model = SASRec(
                num_items=num_items,
                hidden_size=cfg.student.hidden_size,
                num_heads=cfg.student.num_heads,
                num_layers=cfg.student.num_layers,
                dropout_rate=cfg.student.dropout_rate,
                max_seq_len=max_seq_len,
            ).to(device)
        
        # Freeze rec_model parameters
        for param in rec_model.parameters():
            param.requires_grad = False
        print("SASRec model parameters frozen.")
        
        # projectorのインスタンス化
        projector = MLPProjector(
            input_dim=cfg.student.hidden_size, # rec_modelの出力次元
            output_dim=llm.config.hidden_size, # LLMの入力次元
            hidden_size=cfg.teacher.hidden_size,
            dropout_rate=cfg.teacher.dropout_rate
        ).to(device)

        model = iLoRAModel(
            llm=llm, # LLMオブジェクトを渡す
            tokenizer=tokenizer, # Tokenizerオブジェクトを渡す
            num_lora_experts=cfg.teacher.num_lora_experts,
            lora_r=cfg.teacher.lora_r,
            lora_alpha=cfg.teacher.lora_alpha,
            lora_dropout=cfg.teacher.lora_dropout,
            num_items=num_items,
            hidden_size=cfg.teacher.hidden_size,
            dropout_rate=cfg.teacher.dropout_rate,
            rec_model=rec_model, # Pass rec_model
            projector=projector, # Pass projector
            candidate_topk=candidate_topk # Pass candidate_topk
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
    
    # LLMとTokenizerをロード (テスト用)
    llm_test = AutoModelForCausalLM.from_pretrained(cfg.teacher.llm_model_name)
    tokenizer_test = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name)
    if tokenizer_test.pad_token is None:
        tokenizer_test.pad_token = tokenizer_test.eos_token
    
    # 特殊トークンを追加
    tokenizer_test.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm_test.resize_token_embeddings(len(tokenizer_test)) # トークン埋め込み層のサイズを調整

    padding_item_id_dummy = tokenizer_test.pad_token_id # Use tokenizer's pad_token_id

    # ダミーのrec_modelとprojectorを作成
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.item_embeddings = nn.Embedding(num_items_dummy + 1, hidden_size)
            self.cacu_x = lambda x: self.item_embeddings(x)
            self.cacul_h = lambda x, y: torch.randn(x.shape[0], hidden_size)
    
    dummy_rec_model = DummyRecModel(cfg.teacher.hidden_size).to(llm_test.device)
    dummy_projector = MLPProjector(
        input_dim=cfg.teacher.hidden_size,
        output_dim=llm_test.config.hidden_size,
        hidden_size=cfg.teacher.hidden_size,
        dropout_rate=cfg.teacher.dropout_rate
    ).to(llm_test.device)

    # モデルの生成
    teacher_model = create_teacher_model(
        cfg, 
        num_items_dummy, 
        max_seq_len_dummy, 
        dummy_item_id_to_name,
        padding_item_id_dummy,
        rec_model=dummy_rec_model, # Pass dummy rec_model
        projector=dummy_projector # Pass dummy projector
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
