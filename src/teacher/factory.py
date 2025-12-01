from omegaconf import DictConfig
from src.teacher.interfaces import TeacherModel
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector
import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModel, AutoTokenizer
from src.student.models import SASRec
import logging
import os

logger = logging.getLogger(__name__)

def create_teacher_model(cfg: DictConfig, llm_tokenizer: AutoTokenizer, num_items: int, max_seq_len: int, item_id_to_name: Dict[int, str], padding_item_id: int, candidate_topk: int) -> TeacherModel:
    """
    Hydraの設定に基づいて教師モデル（iLoRA）のインスタンスを生成します。
    
    Args:
        cfg: Hydraの設定オブジェクト
        llm_tokenizer: ロード済みのLLMトークナイザー
        num_items: アイテムの総数
        max_seq_len: シーケンスの最大長
        item_id_to_name: アイテムIDから名前へのマッピング辞書
        padding_item_id: パディング用のアイテムID
        candidate_topk: 評価時に考慮する上位候補数
        
    Returns:
        TeacherModel: 初期化された教師モデル
    """
    torch.set_float32_matmul_precision('high')
    model_type = cfg.teacher.model_type

    if model_type == "ilora":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("教師モデル用のLLMをロードしています...")
        
        llm_load_kwargs = {}
        
        if cfg.teacher.get("use_flash_attention", False):
            print("LLMでFlash Attention 2が有効化されました。")
            llm_load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            print("LLMでFlash Attentionは無効化されています。")

        if cfg.teacher.get("use_qlora", False):
            # QLoRAはカスタムMoeLoraModelとの互換性の問題により一時的に無効化されています
            print("QLoRAが設定されていますが、一時的に無効化されています。")

        if cfg.train.get("precision") == "bf16-mixed":
             llm_load_kwargs["torch_dtype"] = torch.bfloat16

        print("DEBUG: from_pretrainedを使用してLLMのロードを試みています...")
        llm = AutoModel.from_pretrained(cfg.teacher.llm_model_name, **llm_load_kwargs)
        logger.info(f"LLMのロードに成功しました: {llm.config._name_or_path}")

        # LLMの語彙サイズを拡張 (アイテムID用)
        original_vocab_size = len(llm_tokenizer)
        # Item IDは1始まりでnum_itemsまであるため、+1しておく (ID 0はパディングだが、マッピングの都合上)
        new_vocab_size = original_vocab_size + num_items + 1
        print(f"LLMの語彙サイズを拡張します: {original_vocab_size} -> {new_vocab_size} (+{num_items + 1} items)")
        llm.resize_token_embeddings(new_vocab_size)

        # アイテムEmbeddingの初期化
        if cfg.teacher.get("rec_model_checkpoint_path"):
            print(f"{cfg.teacher.rec_model_checkpoint_path} から事前学習済みSASRecモデルをロードして初期化に使います")
            rec_model = SASRec(
                num_items=num_items,
                hidden_size=cfg.student.hidden_size,
                num_heads=cfg.student.num_heads,
                num_layers=cfg.student.num_layers,
                dropout_rate=cfg.student.dropout_rate,
                max_seq_len=max_seq_len,
                padding_item_id=padding_item_id,
            )
            checkpoint = torch.load(cfg.teacher.rec_model_checkpoint_path, map_location='cpu')
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    new_state_dict[k[len('model.'):]] = v
                else:
                    new_state_dict[k] = v
            rec_model.load_state_dict(new_state_dict)
            
            # SASRecのEmbeddingをLLMの拡張部分にコピー
            # SASRecのEmbedding次元(hidden_size)とLLMの次元が違う場合、射影が必要か、
            # あるいはパディングで埋めるか。通常は次元が違うので、ここではランダム初期化 + 射影学習の方が良いが、
            # E4SRecの論文ではどうしているか？ -> 通常はLinear層で変換してから足すか、LoRAで吸収する。
            # 今回はシンプルに、SASRecを使わずランダム初期化（LoRAに任せる）か、
            # もし次元が同じならコピーする。
            # Anyo設定: Student(SASRec)=64, LLM=2048 (Qwen1.5-1.8B) -> 次元が違う！
            # コピーできないので、SASRecからの初期化は諦めてランダム初期化にします。
            print("注意: SASRecとLLMの隠れ層次元が異なるため、SASRecからの重みコピーはスキップし、ランダム初期化を使用します。")
            del rec_model
        else:
            print("SASRecチェックポイントが指定されていないため、アイテムEmbeddingはランダム初期化されます。")

        # Projectorは不要なので削除

        # SASRecモデルのロード (アンサンブル用)
        sasrec_model = None
        if cfg.teacher.get("sasrec_model_path"):
            print(f"アンサンブル用のSASRecモデルを {cfg.teacher.sasrec_model_path} からロードします...")
            sasrec_model = SASRec(
                num_items=num_items,
                hidden_size=cfg.student.hidden_size,
                num_heads=cfg.student.num_heads,
                num_layers=cfg.student.num_layers,
                dropout_rate=cfg.student.dropout_rate,
                max_seq_len=max_seq_len,
                padding_item_id=padding_item_id,
            )
            checkpoint = torch.load(cfg.teacher.sasrec_model_path, map_location='cpu')
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    new_state_dict[k[len('model.'):]] = v
                else:
                    new_state_dict[k] = v
            sasrec_model.load_state_dict(new_state_dict)
            
            # Freeze SASRec
            for param in sasrec_model.parameters():
                param.requires_grad = False
            sasrec_model.eval()
            print("SASRecモデルをロードし、凍結しました。")

        model = iLoRAModel(
            llm=llm,
            tokenizer=llm_tokenizer,
            num_lora_experts=cfg.teacher.num_lora_experts,
            lora_r=cfg.teacher.lora_r,
            lora_alpha=cfg.teacher.lora_alpha,
            lora_dropout=cfg.teacher.lora_dropout,
            num_items=num_items,
            hidden_size=cfg.teacher.hidden_size,
            dropout_rate=cfg.teacher.dropout_rate,
            candidate_topk=candidate_topk,
            item_id_to_name=item_id_to_name,
            padding_item_id=padding_item_id,
            llm_dtype=llm.dtype,
            original_vocab_size=original_vocab_size, # アイテムIDのオフセット用
            sasrec_model=sasrec_model # Pass SASRec model
        )

        # パラメータ凍結設定
        # 1. LLM全体を凍結
        for param in llm.parameters():
            param.requires_grad = False
            
        # 2. アイテムEmbedding部分のみ解凍
        # input_embeddingsレイヤーを取得
        input_embeddings = llm.get_input_embeddings()
        input_embeddings.weight.requires_grad = True
        
        # フックを登録して、元の語彙部分の勾配をゼロにする（学習させない）
        def zero_grad_hook(grad):
            # original_vocab_size以前の勾配を0にする
            grad[:original_vocab_size] = 0
            return grad
            
        input_embeddings.weight.register_hook(zero_grad_hook)
        print("アイテムEmbedding部分のみ学習可能に設定しました（元の語彙は凍結）。")

        if cfg.teacher.get("use_torch_compile", False):
            print("torch.compileを使用して教師モデルをコンパイルしています...")
            model = torch.compile(model)
            
        return model
    else:
        raise ValueError(f"Unknown teacher model type: {model_type}")

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 3,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "hidden_size": 64,
            "dropout_rate": 0.1,
            "rec_model_checkpoint_path": "path/to/dummy_checkpoint.ckpt" # Added dummy path
        },
        "student": {
            "hidden_size": 64,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.1,
            "max_seq_len": 50,
        },
        "train": {
            "precision": "32"
        }
    })

    num_items_dummy = 5000
    max_seq_len_dummy = 50
    dummy_item_id_to_name = {i: f"Item {i}" for i in range(num_items_dummy + 1)}
    
    # Create a dummy checkpoint file for rec_model
    dummy_rec_model_state = {
        'state_dict': {f'model.layer_{i}': torch.randn(10, 10) for i in range(2)}
    }
    torch.save(dummy_rec_model_state, cfg.teacher.rec_model_checkpoint_path)

    teacher_model = create_teacher_model(
        cfg, 
        num_items_dummy, 
        max_seq_len_dummy, 
        dummy_item_id_to_name,
        padding_item_id=0,
        candidate_topk=10
    )
    print(f"Created teacher model type: {type(teacher_model)}")
    assert isinstance(teacher_model, TeacherModel)
    assert isinstance(teacher_model, torch.nn.Module)

    # Clean up dummy checkpoint
    import os
    os.remove(cfg.teacher.rec_model_checkpoint_path)

    print("\nTeacher model factory test passed!")

