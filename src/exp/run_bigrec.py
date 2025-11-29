import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import os
import logging
import sys
from pathlib import Path

from src.teacher.bigrec_model import BigRecModel
from src.student.datamodule import SASRecDataModule
from src.data.collators import BigRecCollator
from src.exp.compute_item_embeddings import compute_embeddings

logger = logging.getLogger(__name__)

def run_experiment(cfg: DictConfig):
    pl.seed_everything(cfg.experiment.seed)

    # Validate Config
    if cfg.teacher.get("model_type") != "bigrec":
        logger.warning(f"WARNING: cfg.teacher.model_type is '{cfg.teacher.get('model_type')}', but this script is for BigRec.")
        logger.warning("You are likely loading the wrong config (e.g., 'ilora').")
        logger.warning("Please run with 'experiment=bigrec_movielens' or 'teacher=bigrec'.")
        # We can force continue, but defaults might be wrong.

    # Check and compute item embeddings if needed
    if cfg.teacher.get("compute_item_embeddings", False):
        logger.info("compute_item_embeddings is True. Computing embeddings...")
        compute_embeddings(cfg)
    
    # Log usage of embeddings if path is set
    item_embeddings_path = cfg.teacher.get("item_embeddings_path")
    if item_embeddings_path:
        path_obj = Path(item_embeddings_path)
        if path_obj.exists():
            logger.info(f"Using existing item embeddings at {path_obj}")
        else:
            logger.warning(f"Item embeddings path set to {path_obj} but file not found. Validation will fallback to exact match.")
    
    # 1. DataModule
    dm = SASRecDataModule(conf=cfg.dataset)
    dm.setup()
    
    # 2. Model
    model = BigRecModel(
        model_name_or_path=cfg.teacher.llm_model_name,
        lora_r=cfg.teacher.lora_r,
        lora_alpha=cfg.teacher.lora_alpha,
        lora_dropout=cfg.teacher.lora_dropout,
        learning_rate=cfg.teacher.learning_rate,
        max_source_length=cfg.teacher.max_source_length,
        max_target_length=cfg.teacher.max_target_length,
        item_id_to_name=dm.item_id_to_name,
        metrics_k=cfg.teacher.metrics_k,
        num_beams=cfg.teacher.get("num_beams", 4),
        item_embeddings_path=cfg.teacher.get("item_embeddings_path")
    )
    
    # 3. Custom Collator
    collator = BigRecCollator(
        tokenizer=model.tokenizer,
        item_id_to_name=dm.mapped_id_to_title,
        max_source_length=cfg.teacher.max_source_length,
        max_target_length=cfg.teacher.max_target_length,
        use_cot=cfg.teacher.get("use_cot", False),
        max_history_items=cfg.teacher.get("max_history_items", 20)
    )
    
    # Wrap dataloaders with custom collator
    # Since SASRecDataModule creates DataLoaders internally in train_dataloader(),
    # we might need to override or manually create DataLoaders here.
    # For simplicity, let's manually create them using dm.train_dataset
    
    train_loader = DataLoader(
        dm.train_dataset,
        batch_size=cfg.teacher.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dm.val_dataset,
        batch_size=cfg.teacher.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    # 4. Trainer
    logger_tb = TensorBoardLogger("result", name="bigrec")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="result/bigrec/checkpoints",
        filename="bigrec-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=logger_tb,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        val_check_interval=cfg.teacher.get("val_check_interval", cfg.train.val_check_interval),
        log_every_n_steps=cfg.train.log_every_n_steps
    )
    
    # 5. Fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    run_experiment(cfg)

if __name__ == "__main__":
    main()
