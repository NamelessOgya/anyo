import logging
import torch # Added import
from tqdm import tqdm # tqdmをインポート
from omegaconf import DictConfig, OmegaConf
from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info

from src.student.datamodule import SASRecDataModule # 教師モデルも同じデータモジュールを使用
from src.teacher.factory import create_teacher_model
from src.teacher.mlp_projector import MLPProjector # Added import
import hydra # hydraをインポート

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def run_teacher(cfg: DictConfig):
    # 1. ロギング、シード、Git情報の初期化
    output_dir = get_project_root() / "result" / cfg.run.dir.split('/')[-1]
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists
    
    # Save the Hydra config to the experiment directory
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    setup_logging(log_dir=output_dir / "logs")
    set_seed(cfg.seed)
    git_info = get_git_info()
    logger.info(f"Git Info: {git_info}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # 2. SASRecDataModuleのインスタンス化とデータ準備
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len, # 教師モデルも同じデータモジュールを使用
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # rec_modelのロード
    from src.student.models import SASRec # 適切なモデルをインポート
    rec_model_instance = SASRec(
        num_items=dm.num_items,
        hidden_size=cfg.student.hidden_size,
        num_heads=cfg.student.num_heads,
        num_layers=cfg.student.num_layers,
        dropout_rate=cfg.student.dropout_rate,
        max_seq_len=cfg.student.max_seq_len,
    )
    
    # projectorのインスタンス化
    projector_instance = MLPProjector(
        input_dim=cfg.student.hidden_size, # rec_modelの出力次元
        output_dim=cfg.teacher.hidden_size, # LLMの入力次元 (暫定、ilora_model内でLLMロード後に取得)
        hidden_size=cfg.teacher.hidden_size,
        dropout_rate=cfg.teacher.dropout_rate
    )

    # 3. create_teacher_model を使用して iLoRAModel をインスタンス化
    ilora_model_instance = create_teacher_model(
        cfg,
        num_items=dm.num_items,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=dm.item_id_to_name,
        padding_item_id=dm.padding_item_id,
        candidate_topk=cfg.distill.candidate_topk
    )

    # 4. オプティマイザと損失関数の定義
    optimizer = torch.optim.AdamW(ilora_model_instance.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5. 手動での学習ループ
    logger.info("Starting iLoRA teacher model training...")
    device = "cuda" if torch.cuda.is_available() and cfg.train.accelerator == "gpu" else "cpu"
    ilora_model_instance.to(device)

    best_val_loss = float('inf')
    best_model_path = output_dir / "checkpoints" / "best_teacher_model.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.train.max_epochs):
        # Training loop
        ilora_model_instance.train()
        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(dm.train_dataloader(), desc=f"Epoch {epoch+1} Training")):
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            optimizer.zero_grad()
            
            outputs = ilora_model_instance(batch)
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            logits = ilora_model_instance.item_prediction_head(last_hidden_state)
            next_item = batch["next_item"].squeeze(-1)
            
            loss = loss_fn(logits, next_item)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # if (batch_idx + 1) % cfg.train.log_every_n_steps == 0:
            #     logger.info(f"Epoch {epoch+1}/{cfg.train.max_epochs} | Batch {batch_idx+1}/{len(dm.train_dataloader())} | Train Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(dm.train_dataloader())
        logger.info(f"Epoch {epoch+1}/{cfg.train.max_epochs} - Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        ilora_model_instance.eval()
        val_loss = 0
        all_predictions = []
        all_ground_truths = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dm.val_dataloader(), desc=f"Epoch {epoch+1} Validation")):
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                outputs = ilora_model_instance(batch)
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                logits = ilora_model_instance.item_prediction_head(last_hidden_state)
                next_item = batch["next_item"].squeeze(-1)
                
                loss = loss_fn(logits, next_item)
                val_loss += loss.item()

                _, predicted_indices = torch.topk(logits, k=cfg.eval.metrics_k, dim=1)
                all_predictions.extend(predicted_indices.cpu().tolist())
                all_ground_truths.extend([[item_id.item()] for item_id in next_item.cpu()])

                # if (batch_idx + 1) % cfg.train.log_every_n_steps == 0:
                #     logger.info(f"Epoch {epoch+1}/{cfg.train.max_epochs} | Validating Batch {batch_idx+1}/{len(dm.val_dataloader())}")

        avg_val_loss = val_loss / len(dm.val_dataloader())
        
        from src.core.metrics import calculate_metrics
        metrics = calculate_metrics(all_predictions, all_ground_truths, k=cfg.eval.metrics_k)
        
        logger.info(f"Epoch {epoch+1}/{cfg.train.max_epochs} - Val Loss: {avg_val_loss:.4f}, Val Recall@{cfg.eval.metrics_k}: {metrics.get('recall@k', 0):.4f}, Val NDCG@{cfg.eval.metrics_k}: {metrics.get('ndcg@k', 0):.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ilora_model_instance.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")

    logger.info("iLoRA teacher model training finished.")

    # 7. 評価の実行
    logger.info("Starting iLoRA teacher model evaluation on test set...")
    if best_model_path.exists():
        logger.info(f"Loading best teacher model from {best_model_path} for evaluation.")
        ilora_model_instance.load_state_dict(torch.load(best_model_path))
    else:
        logger.warning("No best teacher model checkpoint found. Using final model for evaluation.")

    ilora_model_instance.eval()
    test_loss = 0
    all_predictions = []
    all_ground_truths = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            outputs = ilora_model_instance(batch)
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            logits = ilora_model_instance.item_prediction_head(last_hidden_state)
            next_item = batch["next_item"].squeeze(-1)
            
            loss = loss_fn(logits, next_item)
            test_loss += loss.item()

            _, predicted_indices = torch.topk(logits, k=cfg.eval.metrics_k, dim=1)
            all_predictions.extend(predicted_indices.cpu().tolist())
            all_ground_truths.extend([[item_id.item()] for item_id in next_item.cpu()])

    avg_test_loss = test_loss / len(dm.test_dataloader())
    metrics = calculate_metrics(all_predictions, all_ground_truths, k=cfg.eval.metrics_k)
    logger.info(f"Test Loss: {avg_test_loss:.4f}, Test Recall@{cfg.eval.metrics_k}: {metrics.get('recall@k', 0):.4f}, Test NDCG@{cfg.eval.metrics_k}: {metrics.get('ndcg@k', 0):.4f}")
    logger.info("iLoRA teacher model evaluation finished.")

    # 8. 訓練データセットに対する教師出力を生成し、保存
    logger.info("Generating teacher outputs for the training dataset...")
    
    teacher_outputs_batches_dir = output_dir / "teacher_outputs_batches"
    teacher_outputs_batches_dir.mkdir(parents=True, exist_ok=True)

    ilora_model_instance.eval()
    batch_idx = 0
    with torch.no_grad():
        for batch in dm.train_dataloader():
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            teacher_outputs = ilora_model_instance.get_teacher_outputs(batch)
            
            # Save each batch's output to a separate file
            batch_output_path = teacher_outputs_batches_dir / f"batch_{batch_idx:05d}.pt"
            torch.save({
                "ranking_scores": teacher_outputs["ranking_scores"].cpu(),
                "embeddings": teacher_outputs["embeddings"].cpu(),
                "candidates": teacher_outputs["candidates"].cpu(),
                "confidence": teacher_outputs["confidence"].cpu(),
            }, batch_output_path)
            batch_idx += 1
            
    logger.info(f"Teacher outputs for {batch_idx} batches saved to {teacher_outputs_batches_dir}")

if __name__ == "__main__":
    run_teacher()