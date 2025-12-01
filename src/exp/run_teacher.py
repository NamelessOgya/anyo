from src.core.config_utils import load_hydra_config
import logging
import time
import os
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
from src.core.logger import setup_logging
from src.core.git_info import get_git_info
# from src.core.callbacks import CustomRichProgressBar # Removed

from src.student.datamodule import SASRecDataModule
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def run_experiment(cfg):
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
    llm_tokenizer.pad_token_id = 0 # Reference uses 0 (unk)
    llm_tokenizer.pad_token = llm_tokenizer.decode(0)
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm_tokenizer.padding_side = "left"

    # 2. SASRecDataModuleのインスタンス化とデータ準備
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.teacher.get("batch_size", cfg.train.batch_size),
        max_seq_len=cfg.teacher.get("max_seq_len", cfg.student.max_seq_len),
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

    # 3. Determine Model Type and Instantiate
    model_type = cfg.teacher.get("model_type", "ilora")
    logger.info(f"Teacher Model Type: {model_type}")

    # 2.5. Compute Item Embeddings (if needed)
    compute_embeddings = cfg.teacher.get("compute_item_embeddings", False)
    logger.info(f"DEBUG: compute_item_embeddings = {compute_embeddings}")
    if compute_embeddings:
        from src.teacher.embedding_utils import compute_and_save_item_embeddings
        compute_and_save_item_embeddings(
            cfg,
            llm_tokenizer=llm_tokenizer,
            item_id_to_name=dm.mapped_id_to_title,
            num_items=dm.num_items
        )

    if model_type in ["bigrec", "moe_bigrec"]:
        from src.data.collators import BigRecCollator
        
        # Instantiate Model via Factory
        model = create_teacher_model(
            cfg,
            llm_tokenizer=llm_tokenizer,
            num_items=dm.num_items,
            max_seq_len=cfg.student.max_seq_len,
            item_id_to_name=dm.mapped_id_to_title,
            padding_item_id=dm.padding_item_id,
            candidate_topk=cfg.distill.candidate_topk
        )
        
        # Custom Collator for BIGRec / MoE-BIGRec
        collator = BigRecCollator(
            tokenizer=model.tokenizer,
            item_id_to_name=dm.mapped_id_to_title,
            max_source_length=cfg.teacher.max_source_length,
            max_target_length=cfg.teacher.max_target_length,
            use_cot=cfg.teacher.get("use_cot", False),
            train_on_inputs=cfg.teacher.get("train_on_inputs", True), # Default to True to match reference
            max_history_items=cfg.teacher.get("max_history_items", 20),
            sasrec_max_seq_len=cfg.student.max_seq_len
        )
        
        # Override DataLoaders with custom collator
        # Note: SASRecDataModule uses default_collate or None by default.
        # We need to recreate dataloaders or use a wrapper.
        # Since dm.train_dataloader() creates new DL every time, we can't just set collate_fn on dm.
        # But we can pass the collator to the Trainer if we were using a custom loop, but PL uses dm.
        # SASRecDataModule doesn't easily support dynamic collator injection via init unless we change it.
        # Workaround: Create DataLoaders manually here and pass to trainer.fit
        
        train_loader = DataLoader(
            dm.train_dataset,
            batch_size=cfg.teacher.get("batch_size", cfg.train.batch_size),
            shuffle=True,
            num_workers=cfg.train.num_workers,
            collate_fn=collator,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dm.val_dataset,
            batch_size=cfg.teacher.get("batch_size", cfg.train.batch_size),
            shuffle=False,
            num_workers=cfg.train.num_workers,
            collate_fn=collator,
            pin_memory=True
        )
        
        trainer_model = model # For fit
        
    else:
        # iLoRA Logic
        ilora_model_instance = create_teacher_model(
            cfg,
            llm_tokenizer=llm_tokenizer,
            num_items=dm.num_items,
            max_seq_len=cfg.student.max_seq_len,
            item_id_to_name=dm.mapped_id_to_title,
            padding_item_id=dm.padding_item_id,
            candidate_topk=cfg.distill.candidate_topk
        )

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

    # 5. PyTorch Lightning Trainer Setup
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs", version="")
    
    # Checkpoint filename differs slightly
    if model_type in ["bigrec", "moe_bigrec"]:
        filename = f"{model_type}-{{epoch:02d}}-{{val_loss:.2f}}"
        monitor = "val_loss"
        mode = "min"
    else:
        filename = "teacher-{epoch:02d}-{val_hr@10:.4f}"
        monitor = "val_hr@10"
        mode = "max"

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=50, process_position=0)
    
    callbacks = [checkpoint_callback, lr_monitor, progress_bar]
    
    if model_type == "ilora":
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_hr@10",
            mode="max",
            patience=5,
            verbose=True
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        default_root_dir=os.getcwd(),
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.teacher.get("num_epochs", cfg.train.max_epochs),
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.teacher.get("accumulate_grad_batches", cfg.train.accumulate_grad_batches),
        val_check_interval=cfg.teacher.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        # strategy="ddp_find_unused_parameters_true" if cfg.train.devices > 1 else "auto"
    )
    logger.info(f"Starting {model_type} teacher model training...")
    training_start_time = time.perf_counter()
    
    if model_type in ["bigrec", "moe_bigrec"]:
        trainer.fit(trainer_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(trainer_model, datamodule=dm)
        
    training_duration = time.perf_counter() - training_start_time
    logger.info(f"{model_type} teacher model training finished. Duration: {training_duration:.2f} seconds.")

    # 6. Evaluation (Only for iLoRA for now, BIGRec evaluation is separate inference)
    if model_type == "ilora":
        logger.info("Starting iLoRA teacher model testing...")
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and Path(best_model_path).exists():
            logger.info(f"Loading best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=trainer_model.device)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            trainer_model.load_state_dict(state_dict, strict=False)
            trainer.test(model=trainer_model, datamodule=dm, ckpt_path=None)
        else:
            trainer.test(model=trainer_model, datamodule=dm)
    
    logger.info(f"Teacher run finished. Results are in: {output_dir}")

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

def main():
    if "--help" in sys.argv:
        print("Usage: python run_teacher.py [overrides]")
        sys.exit(0)

    # --- Centralized Hydra Initialization ---
    overrides = sys.argv[1:]
    cfg = load_hydra_config(config_path="../../conf", overrides=overrides)
    run_experiment(cfg)

if __name__ == "__main__":
    main()