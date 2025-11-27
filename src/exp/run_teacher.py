from src.core.config_utils import load_hydra_config
import logging
import time
from pathlib import Path
from datetime import datetime
import sys
import torch
import shutil
import subprocess
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info
# from src.core.callbacks import CustomRichProgressBar # Removed

from src.student.datamodule import SASRecDataModule
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def main():
    overrides = sys.argv[1:]
    cfg = load_hydra_config(config_path="../../conf", overrides=overrides)

    output_dir = Path(cfg.run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"!!! SCRIPT RUNNING. OUTPUT DIR: {output_dir} !!!")
    
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
        seed=cfg.seed,
        subset_indices_path=cfg.teacher.get("subset_indices_path") # Pass subset indices for Active Learning
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
        item_id_to_name=dm.mapped_id_to_title,
        distill_lambda=cfg.teacher.get("distill_lambda", 0.0),
        distill_loss_type=cfg.teacher.get("distill_loss_type", "mse"),
        distill_decay_type=cfg.teacher.get("distill_decay_type", "none"),
        distill_min_lambda=cfg.teacher.get("distill_min_lambda", 0.0),
        distill_decay_steps=cfg.teacher.get("distill_decay_steps", None)
    )

    # 5. PyTorch Lightning Trainerのセットアップ
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs", version="")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="teacher-{epoch:02d}-{val_hr@10:.4f}",
        monitor="val_hr@10",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Use process_position=0 to attempt to fix multiple bars in Colab
    progress_bar = TQDMProgressBar(refresh_rate=50, process_position=0)
    
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_hr@10",
        mode="max",
        patience=5, # 5 validation checks (2.5 epochs) without improvement
        verbose=True
    )

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar, early_stopping],
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

    # 6. 評価
    logger.info("Starting iLoRA teacher model testing...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and Path(best_model_path).exists():
        logger.info(f"Loading best model from: {best_model_path}")
        # Manually load state_dict with strict=False because we removed frozen weights from checkpoint
        checkpoint = torch.load(best_model_path, map_location=trainer_model.device)
        # Handle both raw state_dict and PL checkpoint format
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        missing_keys, unexpected_keys = trainer_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Model loaded. Missing keys (expected for frozen params): {len(missing_keys)}. Unexpected keys: {len(unexpected_keys)}")
        
        # Run test with the loaded model (ckpt_path=None to prevent reloading)
        trainer.test(model=trainer_model, datamodule=dm, ckpt_path=None)
    else:
        logger.warning("No best model found. Testing with the last model.")
        trainer.test(model=trainer_model, datamodule=dm)
    
    logger.info(f"iLoRA teacher run finished. Results are in: {output_dir}")

    upload_results(cfg, output_dir)

def upload_results(cfg, output_dir):
    # 7. 結果のアップロード（オプション）
    if cfg.get("upload_path"):
        upload_path = Path(cfg.upload_path)
        logger.info(f"Uploading results to: {upload_path}")
        try:
            # Create destination directory if it doesn't exist
            upload_path.mkdir(parents=True, exist_ok=True)
            
            # Use cp -R for faster copy than shutil.copytree
            # Copy contents of output_dir to upload_path
            command = f"cp -R {output_dir}/* {upload_path}/"
            logger.info(f"Executing: {command}")
            
            # subprocess.run with shell=True to handle wildcards
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            
            logger.info("Upload completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upload results (cp command failed): {e.stderr}")
        except Exception as e:
            logger.error(f"Failed to upload results: {e}")

if __name__ == "__main__":
    main()