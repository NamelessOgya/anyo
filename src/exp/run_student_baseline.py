import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info

from src.student.datamodule import SASRecDataModule
from src.student.trainer_baseline import SASRecTrainer
from src.student.evaluator import SASRecEvaluator

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def run_student_baseline(cfg: DictConfig):
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
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 3. SASRecTrainerのインスタンス化
    trainer_model = SASRecTrainer(
        num_items=dm.num_items,
        hidden_size=cfg.student.hidden_size,
        num_heads=cfg.student.num_heads,
        num_layers=cfg.student.num_layers,
        dropout_rate=cfg.student.dropout_rate,
        max_seq_len=cfg.student.max_seq_len,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k
    )

    # 4. PyTorch Lightning Trainerのインスタンス化と学習の実行
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best_model",
        monitor=f"val_recall@{cfg.eval.metrics_k}",
        mode="max",
        save_top_k=1,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard_logs")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        enable_checkpointing=True,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_progress_bar=False
    )

    logger.info("Starting student baseline training...")
    trainer.fit(trainer_model, dm.train_dataloader(), dm.val_dataloader())
    logger.info("Student baseline training finished.")

    # 5. 学習済みモデルの保存 (ベストモデルはModelCheckpointで保存される)
    # ここでは、最終モデルも保存する
    final_model_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # 6. 評価の実行
    logger.info("Starting student baseline evaluation on test set...")
    # ベストモデルをロードして評価
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path} for evaluation.")
        loaded_model = SASRecTrainer.load_from_checkpoint(
            best_model_path,
            num_items=dm.num_items,
            hidden_size=cfg.student.hidden_size,
            num_heads=cfg.student.num_heads,
            num_layers=cfg.student.num_layers,
            dropout_rate=cfg.student.dropout_rate,
            max_seq_len=cfg.student.max_seq_len,
            learning_rate=cfg.train.learning_rate, # ダミー値
            weight_decay=cfg.train.weight_decay, # ダミー値
            metrics_k=cfg.eval.metrics_k
        )
    else:
        logger.warning("No best model checkpoint found. Using final model for evaluation.")
        loaded_model = trainer_model # ベストモデルがない場合は最終モデルを使用

    evaluator = SASRecEvaluator(loaded_model.model, dm, metrics_k=cfg.eval.metrics_k)
    test_metrics = evaluator.evaluate(dm.test_dataloader())
    logger.info(f"Test Metrics: {test_metrics}")

    # 結果を保存
    with open(output_dir / "test_metrics.txt", "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Test metrics saved to {output_dir / 'test_metrics.txt'}")

if __name__ == "__main__":
    run_student_baseline()
