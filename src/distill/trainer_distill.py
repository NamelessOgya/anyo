import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from typing import Dict, Any, Optional

from src.student.models import SASRec
from src.teacher.interfaces import TeacherModel
from src.distill.kd_losses import RankingDistillationLoss, EmbeddingDistillationLoss
from src.distill.selection_policy import SelectionPolicy, AllSamplesPolicy
from src.distill.data_bridge import DataBridge
from src.core.metrics import calculate_metrics

class DistillationTrainer(pl.LightningModule):
    def __init__(self,
                 student_model: SASRec,
                 teacher_model: TeacherModel,
                 num_items: int,
                 ranking_loss_weight: float = 1.0,
                 embedding_loss_weight: float = 1.0,
                 ce_loss_weight: float = 1.0,
                 ranking_temperature: float = 1.0,
                 embedding_loss_type: str = "mse",
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01,
                 metrics_k: int = 10,
                 selection_policy: Optional[SelectionPolicy] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['student_model', 'teacher_model'])

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval() # 教師モデルは学習しない

        self.num_items = num_items

        # 損失関数
        self.ranking_kd_loss_fn = RankingDistillationLoss(temperature=ranking_temperature)
        self.embedding_kd_loss_fn = EmbeddingDistillationLoss(loss_type=embedding_loss_type)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0) # パディングID=0は損失計算から除外

        # 損失の重み
        self.ranking_loss_weight = ranking_loss_weight
        self.embedding_loss_weight = embedding_loss_weight
        self.ce_loss_weight = ce_loss_weight

        self.metrics_k = metrics_k
        self.selection_policy = selection_policy if selection_policy is not None else AllSamplesPolicy()
        self.data_bridge = DataBridge(num_items=num_items)

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        生徒モデルのフォワードパスと、蒸留に必要な生徒モデルの出力を返します。
        """
        student_logits = self.student_model.predict(item_seq, item_seq_len)
        student_embeddings = self.student_model(item_seq, item_seq_len) # 最後のアイテム表現

        return {
            "logits": student_logits,
            "embeddings": student_embeddings
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        # 教師モデルの出力
        with torch.no_grad():
            teacher_outputs_raw = self.teacher_model.get_teacher_outputs(item_seq, item_seq_len)
            teacher_outputs = self.data_bridge.process_teacher_outputs(teacher_outputs_raw)

        # 生徒モデルの出力
        student_outputs_raw = self.forward(item_seq, item_seq_len)
        student_outputs = self.data_bridge.process_student_outputs(student_outputs_raw)

        # サンプル選択ポリシーの適用
        selected_data = self.selection_policy.select_samples(
            teacher_outputs, student_outputs, batch
        )

        # 損失計算
        total_loss = 0.0

        # 1. ランキング蒸留損失
        ranking_kd_loss = self.ranking_kd_loss_fn(
            selected_data["student_logits"], selected_data["teacher_ranking_scores"]
        )
        total_loss += self.ranking_loss_weight * ranking_kd_loss
        self.log("train_ranking_kd_loss", ranking_kd_loss, on_step=True, on_epoch=True)

        # 2. 埋め込み蒸留損失
        embedding_kd_loss = self.embedding_kd_loss_fn(
            selected_data["student_embeddings"], selected_data["teacher_embeddings"]
        )
        total_loss += self.embedding_loss_weight * embedding_kd_loss
        self.log("train_embedding_kd_loss", embedding_kd_loss, on_step=True, on_epoch=True)

        # 3. 通常の推薦損失 (Cross-Entropy)
        ce_loss = self.ce_loss_fn(selected_data["student_logits"], selected_data["next_item"])
        total_loss += self.ce_loss_weight * ce_loss
        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True)

        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        # 生徒モデルの出力
        student_outputs_raw = self.forward(item_seq, item_seq_len)
        student_outputs = self.data_bridge.process_student_outputs(student_outputs_raw)

        # 損失計算 (CE Lossのみで評価)
        ce_loss = self.ce_loss_fn(student_outputs["student_logits"], next_item)
        self.log("val_ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)

        # メトリクス計算
        _, predicted_indices = torch.topk(student_outputs["student_logits"], self.metrics_k, dim=-1)
        ground_truths = [[item.item()] for item in next_item]
        predictions = predicted_indices.tolist()

        metrics = calculate_metrics(predictions, ground_truths, self.metrics_k)
        self.log(f"val_recall@{self.metrics_k}", metrics[f"recall@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val_ndcg@{self.metrics_k}", metrics[f"ndcg@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val_hit_ratio@{self.metrics_k}", metrics[f"hit_ratio@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True)
        
        return {"val_loss": ce_loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        # 生徒モデルの出力
        student_outputs_raw = self.forward(item_seq, item_seq_len)
        student_outputs = self.data_bridge.process_student_outputs(student_outputs_raw)

        # 損失計算 (CE Lossのみで評価)
        ce_loss = self.ce_loss_fn(student_outputs["student_logits"], next_item)
        self.log("test_ce_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)

        # メトリクス計算
        _, predicted_indices = torch.topk(student_outputs["student_logits"], self.metrics_k, dim=-1)
        ground_truths = [[item.item()] for item in next_item]
        predictions = predicted_indices.tolist()

        metrics = calculate_metrics(predictions, ground_truths, self.metrics_k)
        self.log(f"test_recall@{self.metrics_k}", metrics[f"recall@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test_ndcg@{self.metrics_k}", metrics[f"ndcg@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test_hit_ratio@{self.metrics_k}", metrics[f"hit_ratio@{self.metrics_k}"], on_step=False, on_epoch=True, prog_bar=True)
        
        return {"test_loss": ce_loss}

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(self.student_model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

if __name__ == "__main__":
    # テスト用のダミーデータとデータモジュール
    from src.student.datamodule import SASRecDataModule
    from src.teacher.ilora_model import iLoRAModel
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor
    from omegaconf import OmegaConf

    # ダミーデータモジュール
    dm = SASRecDataModule(batch_size=4, max_seq_len=50, num_workers=0)
    dm.prepare_data()
    dm.setup()

    # 生徒モデルのインスタンス化
    student_model_instance = SASRec(
        num_users=1000, # ダミー
        num_items=dm.num_items,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=50
    )

    # 教師モデルのインスタンス化 (ダミー設定)
    teacher_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })
    teacher_model_instance = iLoRAModel(
        llm_model_name=teacher_cfg.llm_model_name,
        num_lora_experts=teacher_cfg.num_lora_experts,
        lora_r=teacher_cfg.lora_r,
        lora_alpha=teacher_cfg.lora_alpha,
        lora_dropout=teacher_cfg.lora_dropout,
        num_items=dm.num_items,
        max_seq_len=dm.max_seq_len,
        hidden_size=teacher_cfg.hidden_size,
        dropout_rate=teacher_cfg.dropout_rate
    )

    # 蒸留トレーナーのインスタンス化
    distill_trainer_model = DistillationTrainer(
        student_model=student_model_instance,
        teacher_model=teacher_model_instance,
        num_items=dm.num_items,
        ranking_loss_weight=1.0,
        embedding_loss_weight=1.0,
        ce_loss_weight=1.0,
        ranking_temperature=2.0,
        embedding_loss_type="mse",
        learning_rate=1e-3,
        weight_decay=0.01,
        metrics_k=10,
        selection_policy=AllSamplesPolicy()
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

    print("\n--- Starting dummy distillation training ---")
    trainer.fit(distill_trainer_model, dm.train_dataloader(), dm.val_dataloader())
    print("Dummy distillation training finished.")

    print("\n--- Starting dummy distillation testing ---")
    trainer.test(distill_trainer_model, dm.test_dataloader())
    print("Dummy distillation testing finished.")
