import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from datetime import datetime
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info
from src.core.callbacks import CustomRichProgressBar

from src.student.datamodule import SASRecDataModule # 教師モデルも同じデータモジュールを使用
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from src.student.evaluator import SASRecEvaluator # 評価は生徒モデルの評価器を流用
from src.teacher.mlp_projector import MLPProjector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch

logger = logging.getLogger(__name__)

def main():
    # --- Manual Hydra Initialization ---
    # Parse command-line overrides manually
    overrides = [arg for arg in sys.argv[1:] if arg.startswith("teacher.")]
    
    with hydra.initialize(config_path="../../conf", version_base="1.3", job_name="teacher_run"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    # --- End Manual Hydra Initialization ---

    # 1. ロギング、シード、Git情報の初期化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_project_root() / "result" / f"teacher_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"!!! SCRIPT RUNNING. MANUALLY CREATED OUTPUT DIR: {output_dir} !!!")
    
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
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 3. create_teacher_model を使用して iLoRAModel をインスタンス化
    ilora_model_instance = create_teacher_model(
        cfg,
        num_items=dm.num_items,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=dm.item_id_to_name,
        padding_item_id=dm.padding_item_id,
        candidate_topk=cfg.distill.candidate_topk
    )

    # 4. iLoRATrainerのインスタンス化
    trainer_model = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k,
        item_id_to_name=dm.item_id_to_name
    )

    # 5. PyTorch Lightning Trainerのセットアップ
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs", version="")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="teacher-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = CustomRichProgressBar()

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        precision="16-mixed"
    )

    logger.info("Starting iLoRA teacher model training...")
    trainer.fit(trainer_model, datamodule=dm)
    logger.info("iLoRA teacher model training finished.")

    # 6. 評価と教師出力の生成
    logger.info("Starting final evaluation and teacher output generation...")

    # 6.1. テストセットでの評価
    # pl.Trainerは、最適なチェックポイントが利用可能な場合、自動的にそれを使用します。
    logger.info("Evaluating on test set with the best model...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and Path(best_model_path).exists():
        trainer.test(model=trainer_model, datamodule=dm, ckpt_path=best_model_path)
    else:
        logger.warning("No best model found. Testing with the last model state.")
        trainer.test(model=trainer_model, datamodule=dm)
    logger.info("Test set evaluation finished.")

    # 6.2. 訓練データセットに対する教師出力を生成し、保存
    # 手動ループのために、モデル全体を正しいデバイスに移動させ、評価モードに設定します。
    logger.info("Preparing for teacher output generation...")
    device = "cuda" if torch.cuda.is_available() and cfg.train.accelerator == "gpu" else "cpu"
    trainer_model.to(device)
    trainer_model.eval()

    logger.info("Generating teacher outputs for the training dataset...")
    teacher_outputs_batches_dir = output_dir / "teacher_outputs_batches"
    teacher_outputs_batches_dir.mkdir(parents=True, exist_ok=True)

    batch_idx = 0
    with torch.no_grad():
        for batch in dm.train_dataloader():
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            teacher_outputs = trainer_model.model.get_teacher_outputs(batch)
            
            # Save each batch's output to a separate file
            batch_output_path = teacher_outputs_batches_dir / f"batch_{batch_idx:05d}.pt"
            torch.save({
                "ranking_scores": teacher_outputs["ranking_scores"].cpu(),
                "embeddings": teacher_outputs["embeddings"].cpu(),
                "candidates": teacher_outputs["candidates"].cpu(),
                "confidence": teacher_outputs["confidence"].cpu(),
            }, batch_output_path)
            batch_idx += 1
            if batch_idx % 10 == 0:
                logger.info(f"Generated outputs for {batch_idx} batches...")
            
    logger.info(f"Teacher outputs for {batch_idx} batches saved to {teacher_outputs_batches_dir}")

    logger.info(f"Teacher model run finished. Results are in: {output_dir}")

if __name__ == "__main__":
    main()