import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.core.callbacks import CustomRichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# ... (omitted for brevity)

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
        max_seq_len=cfg.student.max_seq_len, # 教師モデルも同じmax_seq_lenを使用
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
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