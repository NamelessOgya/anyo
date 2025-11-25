import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from typing import Dict, Any

from src.student.models import SASRec
from src.core.metrics import calculate_metrics

class SASRecTrainer(pl.LightningModule):
    def __init__(self, 
                 num_items: int, 
                 hidden_size: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout_rate: float, 
                 max_seq_len: int,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01,
                 metrics_k: int = 10):
        super().__init__()
        self.save_hyperparameters()

        self.model = SASRec(
            num_items=num_items,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            max_seq_len=max_seq_len
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0) # パディングID=0は損失計算から除外
        self.metrics_k = metrics_k

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        return self.model.predict(item_seq, item_seq_len)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        item_seq = batch["seq"]
        item_seq_len = batch["len_seq"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        
        loss = self.loss_fn(logits, next_item.squeeze(-1) - 1) # Convert to 0-indexed
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        item_seq = batch["seq"]
        item_seq_len = batch["len_seq"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        loss = self.loss_fn(logits, next_item.squeeze(-1) - 1) # Convert to 0-indexed
        # logitsからトップKのアイテムIDを取得
        _, predicted_indices = torch.topk(logits, self.metrics_k, dim=-1)
        
        # next_itemはスカラーなので、リストのリスト形式に変換
        ground_truths = [[item.item() - 1] for item in next_item] # Convert to 0-indexed
        predictions = predicted_indices.tolist()

        metrics = calculate_metrics(predictions, ground_truths, self.metrics_k)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_recall@{self.metrics_k}", metrics[f"recall@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_ndcg@{self.metrics_k}", metrics[f"ndcg@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_hit_ratio@{self.metrics_k}", metrics[f"hit_ratio@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        item_seq = batch["seq"]
        item_seq_len = batch["len_seq"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        loss = self.loss_fn(logits, next_item.squeeze(-1) - 1) # Convert to 0-indexed

        _, predicted_indices = torch.topk(logits, self.metrics_k, dim=-1)
        ground_truths = [[item.item() - 1] for item in next_item] # Convert to 0-indexed
        predictions = predicted_indices.tolist()

        metrics = calculate_metrics(predictions, ground_truths, self.metrics_k)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_recall@{self.metrics_k}", metrics[f"recall@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
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

    trainer_model = SASRecTrainer(
        num_users=num_users_dummy,
        num_items=num_items_actual,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=50,
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
