from src.core.config_utils import load_hydra_config
from src.core.paths import get_project_root
from src.core.logging import setup_logging
from src.core.seed import set_seed
from src.core.git_info import get_git_info
from omegaconf import OmegaConf
from src.student.datamodule import SASRecDataModule
from src.student.models import SASRec
from src.student.trainer_baseline import SASRecTrainer
from src.core.callbacks import CustomRichProgressBar

import logging
import sys
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

def main():
    # Centralized Hydra initialization
    overrides = sys.argv[1:]
    cfg = load_hydra_config(config_path="../../conf", overrides=overrides)

    # Use cfg.run.dir
    output_dir = Path(cfg.run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"!!! SCRIPT RUNNING. OUTPUT DIR: {output_dir} !!!")
    
    # Save the Hydra config to the experiment directory
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # The setup_logging function will direct logs to a 'logs' subdirectory
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
        max_seq_len=cfg.student.max_seq_len, # 教師モデルも同じmax_seq_lenを使用
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 3. SASRecモデルのインスタンス化
    student_model = SASRec(
        num_items=dm.num_items,
        hidden_size=cfg.student.hidden_size,
        num_heads=cfg.student.num_heads,
        num_layers=cfg.student.num_layers,
        dropout_rate=cfg.student.dropout_rate,
        max_seq_len=cfg.student.max_seq_len,
        padding_item_id=dm.padding_item_id
    )

    # 4. SASRecTrainerのインスタンス化
    trainer_model = SASRecTrainer(
        rec_model=student_model,
        num_items=dm.num_items,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k
    )

    # 4. PyTorch Lightning Trainerのインスタンス化と学習の実行
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs", version="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="student-baseline-{epoch:02d}-{val_hr@10:.4f}",
        monitor="val_hr@10",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # progress_bar = CustomRichProgressBar()

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )

    logger.info("Starting student baseline training...")
    trainer.fit(trainer_model, datamodule=dm)

    logger.info("Starting student baseline testing...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and Path(best_model_path).exists():
        logger.info(f"Loading best model from: {best_model_path}")
        trainer.test(model=trainer_model, datamodule=dm, ckpt_path=best_model_path)
    else:
        logger.warning("No best model found. Testing with the last model.")
        trainer.test(model=trainer_model, datamodule=dm)
    
    logger.info(f"Student baseline run finished. Results are in: {output_dir}")

if __name__ == "__main__":
    main()