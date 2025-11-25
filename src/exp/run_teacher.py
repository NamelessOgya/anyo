from src.core.config_utils import load_hydra_config
import logging
import time
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

from src.student.datamodule import SASRecDataModule
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def main():
    # --- Centralized Hydra Initialization ---
    overrides = sys.argv[1:]
    cfg = load_hydra_config(config_path="../../conf", overrides=overrides)

    # 1. ロギング、シード、Git情報の初期化
    # Use Hydra's output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"!!! SCRIPT RUNNING. HYDRA OUTPUT DIR: {output_dir} !!!")
    
    setup_logging(log_dir=output_dir / "logs")
    set_seed(cfg.seed)
    git_info = get_git_info()
    logger.info(f"Git Info: {git_info}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Initialize tokenizer early
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name, use_fast=False)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm_tokenizer.padding_side = "right"

    # 2. SASRecDataModuleのインスタンス化とデータ準備
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        tokenizer=llm_tokenizer, # Pass tokenizer
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv",
        seed=cfg.seed
    )
    dm.prepare_data()
    dm.setup()

    # 3. create_teacher_model を使用して iLoRAModel をインスタンス化
    ilora_model_instance = create_teacher_model(
        cfg,
        llm_tokenizer=llm_tokenizer,
        num_items=dm.num_items,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=dm.mapped_id_to_title,
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
        item_id_to_name=dm.mapped_id_to_title
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
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches
    )

    logger.info("Starting iLoRA teacher model training...")
    training_start_time = time.perf_counter()
    trainer.fit(trainer_model, datamodule=dm)
    training_duration = time.perf_counter() - training_start_time
    logger.info(f"iLoRA teacher model training finished. Duration for trainer.fit: {training_duration:.2f} seconds.")

    # 6. 評価と教師出力の生成
    # This part remains largely the same but needs to use the centralized hydra config 'cfg'
    # and the created output directory 'output_dir'
    ...

if __name__ == "__main__":
    main()