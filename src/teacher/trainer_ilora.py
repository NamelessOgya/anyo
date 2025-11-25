# coding=utf-8
import logging
import time
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.teacher.ilora_model import iLoRAModel
from src.core.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class iLoRATrainer(pl.LightningModule):
    """
    iLoRAモデルの学習および評価を行うPyTorch Lightning Module。
    """
    def __init__(
        self,
        ilora_model: iLoRAModel,
        num_items: int,
        learning_rate: float,
        weight_decay: float,
        metrics_k: int,
        item_id_to_name: Dict[int, str],
        warmup_steps: int = 0,
        max_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ilora_model"])
        self.model = ilora_model
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics_k = metrics_k
        self.item_id_to_name = item_id_to_name
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        学習ステップのフォワードパス。
        """
        return self.model(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        1回の学習ステップ。
        LLMの出力（Causal LM Loss）を計算し、ログに記録します。
        """
        start_time = time.time()
        
        # モデルのフォワードパス（内部で損失計算が行われる）
        outputs = self.model(batch)
        loss = outputs.loss

        # 損失と学習時間をログに記録
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        duration = time.time() - start_time
        self.log("epoch_duration_seconds", duration, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        検証ステップ。
        iLoRAのランキングスコアを使用して、Hit Ratio (HR) や NDCG などのメトリクスを計算します。
        """
        # 蒸留用出力の取得（ランキングスコアを含む）
        outputs = self.model.get_teacher_outputs(batch)
        ranking_scores = outputs["ranking_scores"] # (batch_size, num_items)
        
        # 正解アイテムのID
        target_items = batch["next_item"] # (batch_size,)

        # メトリクスの計算 (HR@K, NDCG@K)
        # 簡易的な実装: 上位K個のアイテムを取得し、正解が含まれているか確認
        _, topk_indices = torch.topk(ranking_scores, k=self.metrics_k, dim=-1) # (batch_size, k)
        
        hits = torch.zeros(target_items.size(0), device=self.device)
        ndcgs = torch.zeros(target_items.size(0), device=self.device)

        for i in range(target_items.size(0)):
            target = target_items[i]
            if target in topk_indices[i]:
                hits[i] = 1.0
                # NDCG計算: 正解がランクのどこにあるか
                rank = (topk_indices[i] == target).nonzero(as_tuple=True)[0].item()
                ndcgs[i] = 1.0 / torch.log2(torch.tensor(rank + 2.0))

        self.log(f"val_hr@{self.metrics_k}", hits.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_ndcg@{self.metrics_k}", ndcgs.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": 0.0} # ダミーの損失（必要に応じて実装）

    def configure_optimizers(self):
        """
        オプティマイザとスケジューラの設定。
        """
        # 重み減衰の適用（バイアスとLayerNormを除く）
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # 線形ウォームアップ付きのスケジューラ
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # 蒸留用出力の取得（ランキングスコアを含む）
        outputs = self.model.get_teacher_outputs(batch)
        ranking_scores = outputs["ranking_scores"] # (batch_size, num_items)
        
        # 正解アイテムのID
        target_items = batch["next_item"] # (batch_size,)

        # Metric calculation
        _, predicted_indices = torch.topk(ranking_scores, k=self.metrics_k, dim=1)
        predictions = predicted_indices.tolist()
        
        ground_truths_list = []
        for item_id_tensor in target_items:
            item_id = item_id_tensor.item()
            ground_truths_list.append([item_id])

        metrics = calculate_metrics(predictions, ground_truths_list, k=self.metrics_k)
        for metric_name, metric_value in metrics.items():
            self.log(f"test_{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_loss": 0.0} # Dummy loss

    def configure_optimizers(self) -> Any:
        # iLoRAでは、LoRAアダプターとゲーティングネットワークのパラメータのみを学習対象とする
        # LLMのベースモデルのパラメータはフリーズされている
        trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {trainable_params}") # Changed to logger.info
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_train_epoch_start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("GPU Memory: Peak memory stats reset for epoch.")
        self.training_epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            logger.info(f"GPU Memory: Max allocated during epoch: {max_memory:.2f} GB")
        
        epoch_duration = time.time() - self.training_epoch_start_time
        self.log("epoch_duration_seconds", epoch_duration, prog_bar=True, logger=True)
        logger.info(f"Epoch duration: {epoch_duration:.2f} seconds")

if __name__ == "__main__":
    # テスト用のダミーデータとデータモジュール
    from src.student.datamodule import SASRecDataModule # データモジュールは共通
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor
    from transformers import AutoModelForCausalLM, AutoTokenizer # LLMとTokenizerをロードするために追加
    from src.teacher.mlp_projector import MLPProjector # Added import for MLPProjector

    # iLoRAModelのダミー設定
    ilora_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })

    # LLMとTokenizerをロード
    llm = AutoModelForCausalLM.from_pretrained(ilora_cfg.llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(ilora_cfg.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))

    # ダミーデータモジュール
    dm = SASRecDataModule(
        batch_size=4, 
        max_seq_len=50, 
        num_workers=0,
        llm_model_name=ilora_cfg.llm_model_name,
        tokenizer=tokenizer,
        max_gen_length=64
    )
    dm.prepare_data()
    dm.setup()

    # ダミーのrec_modelとprojectorを作成
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size_rec, num_items_rec):
            super().__init__()
            self.item_embeddings = nn.Embedding(num_items_rec + 1, hidden_size_rec)
            self.cacu_x = lambda x: self.item_embeddings(x)
            self.cacul_h = lambda x, y: torch.randn(x.shape[0], hidden_size_rec)
    
    dummy_rec_model = DummyRecModel(ilora_cfg.hidden_size, dm.num_items).to(llm.device)
    dummy_projector = MLPProjector(
        input_dim=ilora_cfg.hidden_size,
        output_dim=llm.config.hidden_size,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate
    ).to(llm.device)

    # iLoRAModelのインスタンス化
    ilora_model_instance = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=ilora_cfg.num_lora_experts,
        lora_r=ilora_cfg.lora_r,
        lora_alpha=ilora_cfg.lora_alpha,
        lora_dropout=ilora_cfg.lora_dropout,
        num_items=dm.num_items,
        max_seq_len=dm.max_seq_len,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate,
        item_id_to_name=dm.item_id_to_name,
        padding_item_id=tokenizer.pad_token_id,
        rec_model=dummy_rec_model,
        projector=dummy_projector,
        candidate_topk=10
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
