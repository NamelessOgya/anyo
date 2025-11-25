# coding=utf-8
import logging
import time
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.student.models import SASRec

logger = logging.getLogger(__name__)


class SASRecTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating baseline recommendation models (e.g., SASRec).
    """
    def __init__(
        self,
        rec_model: SASRec,
        num_items: int,
        learning_rate: float,
        weight_decay: float,
        metrics_k: int,
        warmup_steps: int = 0,
        max_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["rec_model"])
        self.model = rec_model
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics_k = metrics_k
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        # Loss function: Cross Entropy Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass for the training step.
        """
        return self.model.predict(batch["seq"], batch["len_seq"])

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        A single training step.
        Calculates the Cross Entropy Loss between the recommendation model's output (logits)
        and the ground truth item.
        """
        start_time = time.time()
        
        # Model's forward pass
        logits = self(batch) # (batch_size, num_items)
        
        # Ground truth item (1-indexed, convert to 0-indexed)
        target = batch["next_item"] - 1
        
        # Loss calculation
        loss = self.loss_fn(logits, target)

        # Log loss and training time
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        duration = time.time() - start_time
        self.log("epoch_duration_seconds", duration, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        Calculates metrics such as Hit Ratio (HR) and NDCG.
        """
        logits = self(batch)
        target = batch["next_item"] - 1
        loss = self.loss_fn(logits, target)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Metric calculation (HR@K, NDCG@K)
        _, topk_indices = torch.topk(logits, k=self.metrics_k, dim=-1)
        
        hits = torch.zeros(target.size(0), device=self.device)
        ndcgs = torch.zeros(target.size(0), device=self.device)

        for i in range(target.size(0)):
            t = target[i]
            if t in topk_indices[i]:
                hits[i] = 1.0
                rank = (topk_indices[i] == t).nonzero(as_tuple=True)[0].item()
                ndcgs[i] = 1.0 / torch.log2(torch.tensor(rank + 2.0))

        self.log(f"val_hr@{self.metrics_k}", hits.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_ndcg@{self.metrics_k}", ndcgs.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        Calculates metrics such as Hit Ratio (HR) and NDCG.
        """
        logits = self(batch)
        target = batch["next_item"] - 1
        loss = self.loss_fn(logits, target)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_ndcg@{self.metrics_k}", metrics[f"ndcg@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_hit_ratio@{self.metrics_k}", metrics[f"hit_ratio@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_loss": loss}

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

if __name__ == "__main__":
    # テスト用のダミーデータとデータモジュール
    from src.student.datamodule import SASRecDataModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor

    # ダミーデータモジュール
    dm = SASRecDataModule(batch_size=4, max_seq_len=50, num_workers=0) # num_workers=0 for Windows/debugging
    dm.prepare_data()
    dm.setup()

    # トレーナーのインスタンス化
    # num_usersとnum_itemsはdatamoduleから取得
    num_users_dummy = 1000 # SASRecモデルではnum_usersは直接使われないが、引数として渡す
    num_items_actual = dm.num_items

    # SASRecモデルのインスタンス化
    rec_model = SASRec(
        num_items=num_items_actual,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=50
    )

    trainer_model = SASRecTrainer(
        rec_model=rec_model,
        num_items=num_items_actual,
        learning_rate=1e-3,
        weight_decay=0.01,
        metrics_k=10
    )

    # PyTorch Lightning Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu", # テスト用にCPUを使用
        devices=1,
        logger=False, # テスト時はロガーを無効化
        callbacks=[lr_monitor],
        enable_checkpointing=False,
    )

    print("\n--- Starting dummy training ---")
    trainer.fit(trainer_model, dm.train_dataloader(), dm.val_dataloader())
    print("Dummy training finished.")

    print("\n--- Starting dummy testing ---")
    trainer.test(trainer_model, dm.test_dataloader())
    print("Dummy testing finished.")
