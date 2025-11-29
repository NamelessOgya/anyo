import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import os
import logging

from src.teacher.bigrec_model import BigRecModel
from src.student.datamodule import SASRecDataModule
from src.data.collators import BigRecCollator

logger = logging.getLogger(__name__)

def run_experiment(cfg: DictConfig):
    pl.seed_everything(cfg.experiment.seed)
    
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
        max_target_length=cfg.teacher.max_target_length
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
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps
    )
    
    # 5. Fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    run_experiment(cfg)

if __name__ == "__main__":
    main()
