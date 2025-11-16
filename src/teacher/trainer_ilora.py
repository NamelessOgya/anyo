import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from typing import Dict, Any

from src.teacher.ilora_model import iLoRAModel
from src.core.metrics import calculate_metrics

class iLoRATrainer(pl.LightningModule):
    def __init__(self, 
                 ilora_model: iLoRAModel,
                 num_items: int,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 metrics_k: int = 10):
        super().__init__()
        self.save_hyperparameters(ignore=['ilora_model'])

        self.model = ilora_model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0) # パディングID=0は損失計算から除外
        self.metrics_k = metrics_k

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        return self.model(item_seq, item_seq_len)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        
        # iLoRAの学習では、次アイテム予測のCE損失を計算
        loss = self.loss_fn(logits, next_item)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        loss = self.loss_fn(logits, next_item)

        _, predicted_indices = torch.topk(logits, self.metrics_k, dim=-1)
        ground_truths = [[item.item()] for item in next_item]
        predictions = predicted_indices.tolist()

        metrics = calculate_metrics(predictions, ground_truths, self.metrics_k)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_recall@{self.metrics_k}", metrics[f"recall@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_ndcg@{self.metrics_k}", metrics[f"ndcg@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_hit_ratio@{self.metrics_k}", metrics[f"hit_ratio@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        loss = self.loss_fn(logits, next_item)

        _, predicted_indices = torch.topk(logits, self.metrics_k, dim=-1)
        ground_truths = [[item.item()] for item in next_item]
        predictions = predicted_indices.tolist()

        metrics = calculate_metrics(predictions, ground_truths, self.metrics_k)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_recall@{self.metrics_k}", metrics[f"recall@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_ndcg@{self.metrics_k}", metrics[f"ndcg@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_hit_ratio@{self.metrics_k}", metrics[f"hit_ratio@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_loss": loss}

    def configure_optimizers(self) -> Any:
        # iLoRAでは、LoRAアダプターとゲーティングネットワークのパラメータのみを学習対象とする
        # LLMのベースモデルのパラメータはフリーズされている
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

if __name__ == "__main__":
    # テスト用のダミーデータとデータモジュール
    from src.student.datamodule import SASRecDataModule # データモジュールは共通
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor

    # ダミーデータモジュール
    dm = SASRecDataModule(batch_size=4, max_seq_len=50, num_workers=0)
    dm.prepare_data()
    dm.setup()

    # iLoRAModelのインスタンス化 (ダミー設定)
    ilora_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })
    ilora_model_instance = iLoRAModel(
        llm_model_name=ilora_cfg.llm_model_name,
        num_lora_experts=ilora_cfg.num_lora_experts,
        lora_r=ilora_cfg.lora_r,
        lora_alpha=ilora_cfg.lora_alpha,
        lora_dropout=ilora_cfg.lora_dropout,
        num_items=dm.num_items,
        max_seq_len=dm.max_seq_len,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate
    )

    # iLoRATrainerのインスタンス化
    ilora_trainer = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=1e-4,
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

    print("\n--- Starting dummy iLoRA training ---")
    trainer.fit(ilora_trainer, dm.train_dataloader(), dm.val_dataloader())
    print("Dummy iLoRA training finished.")

    print("\n--- Starting dummy iLoRA testing ---")
    trainer.test(ilora_trainer, dm.test_dataloader())
    print("Dummy iLoRA testing finished.")
