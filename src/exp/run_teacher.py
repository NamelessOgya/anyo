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

from src.student.datamodule import SASRecDataModule # 教師モデルも同じデータモジュールを使用
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from src.student.evaluator import SASRecEvaluator # 評価は生徒モデルの評価器を流用

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def run_teacher(cfg: DictConfig):
    # 1. ロギング、シード、Git情報の初期化
    output_dir = get_project_root() / "result" / cfg.hydra.run.dir.split('/')[-1]
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
        max_seq_len=cfg.model.max_seq_len, # 教師モデルも同じmax_seq_lenを使用
        num_workers=cfg.train.num_workers
    )
    dm.prepare_data()
    dm.setup()

    # 3. create_teacher_model を使用して iLoRAModel をインスタンス化
    ilora_model_instance = create_teacher_model(
        cfg, 
        num_items=dm.num_items, 
        max_seq_len=dm.max_seq_len
    )

    # 4. iLoRATrainerのインスタンス化
    trainer_model = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k
    )

    # 5. PyTorch Lightning Trainerのインスタンス化と学習の実行
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best_teacher_model",
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
        log_every_n_steps=cfg.train.log_every_n_steps
    )

    logger.info("Starting iLoRA teacher model training...")
    trainer.fit(trainer_model, dm.train_dataloader(), dm.val_dataloader())
    logger.info("iLoRA teacher model training finished.")

    # 6. 学習済みモデルの保存 (ベストモデルはModelCheckpointで保存される)
    final_model_path = output_dir / "final_teacher_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final teacher model saved to {final_model_path}")

    # 7. 評価の実行
    logger.info("Starting iLoRA teacher model evaluation on test set...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best teacher model from {best_model_path} for evaluation.")
        loaded_model = iLoRATrainer.load_from_checkpoint(
            best_model_path,
            ilora_model=ilora_model_instance, # iLoRAModelインスタンスを渡す必要がある
            num_items=dm.num_items,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            metrics_k=cfg.eval.metrics_k
        )
    else:
        logger.warning("No best teacher model checkpoint found. Using final model for evaluation.")
        loaded_model = trainer_model # ベストモデルがない場合は最終モデルを使用

    # 教師モデルの評価には、生徒モデルの評価器を流用する
    # ただし、SASRecEvaluatorはSASRecTrainerを想定しているので、
    # iLoRATrainerを直接渡すことはできない。
    # ここでは、iLoRATrainerのmodel (iLoRAModel) を直接評価器に渡すか、
    # iLoRATrainer自体を評価器として使うラッパーが必要。
    # 簡略化のため、ここではiLoRATrainerのtest_stepを直接呼び出す
    logger.info("Running test_step for teacher model...")
    trainer.test(loaded_model, dm.test_dataloader())
    logger.info("iLoRA teacher model evaluation finished.")

if __name__ == "__main__":
    run_teacher()
