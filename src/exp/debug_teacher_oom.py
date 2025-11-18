import torch
import logging
from omegaconf import OmegaConf

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.student.datamodule import SASRecDataModule
from src.teacher.factory import create_teacher_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def debug_teacher_oom():
    # 1. Configの準備 (最小限のconfigを直接定義)
    cfg_dict = {
        "seed": 42,
        "dataset": {
            "name": "movielens",
            "data_dir": "data/ml-1m",
            "limit_data_rows": 10000,
        },
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 3,
            "lora_r": 9,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "hidden_size": 64,
            "dropout_rate": 0.1,
            "max_gen_length": 20,
            "rec_model_checkpoint_path": "/workspace/result/result_20251118_100122/checkpoints/best_model.ckpt",
        },
        "student": { # teacher.max_seq_lenで使用するため必要
            "max_seq_len": 50,
            "hidden_size": 64, # rec_modelのhidden_sizeとして必要
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.1,
        },
        "train": {
            "batch_size": 32, # OOMを再現しやすいように小さいバッチサイズ
            "num_workers": 0, # デバッグのためワーカー数を0に
        },
        "distill": { # create_teacher_modelで使用するため必要
            "candidate_topk": 10,
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    # 2. ロギング、シードの初期化
    output_dir = get_project_root() / "debug_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir / "logs")
    set_seed(cfg.seed)
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. SASRecDataModuleのインスタンス化とデータ準備
    llm_for_dm = AutoModelForCausalLM.from_pretrained(cfg.teacher.llm_model_name)
    tokenizer_for_dm = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name)
    if tokenizer_for_dm.pad_token is None:
        tokenizer_for_dm.pad_token = tokenizer_for_dm.eos_token
    tokenizer_for_dm.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm_for_dm.resize_token_embeddings(len(tokenizer_for_dm))

    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 4. iLoRAModelのインスタンス化
    ilora_model_instance = create_teacher_model(
        cfg,
        llm=llm_for_dm,
        tokenizer=tokenizer_for_dm,
        num_items=dm.num_items,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=dm.item_id_to_name,
        padding_item_id=dm.padding_item_id,
        candidate_topk=cfg.distill.candidate_topk
    ).to(device)
    ilora_model_instance.train() # Train mode for backward pass

    # 5. データの取得と単一フォワードパスの実行
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))

    # バッチをデバイスに移動
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict) and "input_ids" in value:
            batch[key]["input_ids"] = value["input_ids"].to(device)
            batch[key]["attention_mask"] = value["attention_mask"].to(device)
            if "labels" in value:
                batch[key]["labels"] = value["labels"].to(device)

    # 6. オプティマイザと損失関数の準備
    optimizer = torch.optim.AdamW(ilora_model_instance.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    logger.info(f"Running a single training step with batch size {cfg.train.batch_size}...")
    try:
        # Forward pass
        outputs = ilora_model_instance(batch)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        logits = ilora_model_instance.item_prediction_head(last_hidden_state)
        next_item = batch["next_item"].squeeze(-1)
        
        # Calculate loss
        loss = loss_fn(logits, next_item)
        logger.info(f"Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        logger.info("Single training step successful (forward + backward + optimizer step)!")
        # メモリ使用量をログに出力
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            logger.info(f"CUDA Memory Allocated: {allocated:.2f} GB")
            logger.info(f"CUDA Memory Reserved: {reserved:.2f} GB")

    except Exception as e:
        logger.error(f"Error during training step: {e}")
        if device.type == 'cuda':
            logger.error(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
            logger.error(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")
        raise

if __name__ == "__main__":
    debug_teacher_oom()
