import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any

from src.student.models import SASRec
from src.teacher.interfaces import TeacherModel
from src.distill.kd_losses import RankingDistillationLoss, EmbeddingDistillationLoss, WeightedBCELoss
from src.distill.selection_policy import SelectionPolicy
from src.core.metrics import calculate_metrics

class DistillationTrainer(pl.LightningModule):
    def __init__(self,
                 student_model: SASRec,
                 teacher_model: TeacherModel,
                 num_items: int,
                 ranking_loss_weight: float,
                 embedding_loss_weight: float,
                 ce_loss_weight: float,
                 ranking_temperature: float,
                 embedding_loss_type: str,
                 learning_rate: float,
                 weight_decay: float,
                 metrics_k: int,
                 selection_policy: SelectionPolicy,
                 gamma_position: float,
                 gamma_confidence: float,
                 gamma_consistency: float,
                 candidate_topk: int,
                 ed_weight: float):
        super().__init__()
        self.student_model_module = student_model # Registered as a submodule
        self.teacher_model = teacher_model
        self.num_items = num_items
        self.ranking_loss_weight = ranking_loss_weight
        self.embedding_loss_weight = embedding_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics_k = metrics_k
        self.selection_policy = selection_policy
        self.gamma_position = gamma_position
        self.gamma_confidence = gamma_confidence
        self.gamma_consistency = gamma_consistency
        self.candidate_topk = candidate_topk
        self.student_model_module.ed_weight = ed_weight
        self.student_model_module.train() # Explicitly set to train mode here

        # 損失関数
        self.ranking_kd_loss_fn = WeightedBCELoss()
        self.embedding_kd_loss_fn = EmbeddingDistillationLoss(loss_type=embedding_loss_type)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()

        # 教師モデルは評価モードに設定し、パラメータをフリーズ
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # save_hyperparameters() は pl.LightningModule の機能
        # self.save_hyperparameters(ignore=['student_model', 'teacher_model', 'selection_policy'])

    def on_train_start(self):
        """
        Called at the beginning of training to ensure student model is in train mode.
        """
        self.student_model_module.train()

    def forward(self, item_seq, item_seq_len):
        return self.student_model_module.predict(item_seq, item_seq_len)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.student_model_module.train()
        item_seq = batch["seq"]
        item_seq_len = batch["len_seq"]
        next_item = batch["next_item"]

        # 1. 教師モデルからソフトターゲットと埋め込みを取得
        with torch.no_grad():
            teacher_outputs_raw = self.teacher_model.get_teacher_outputs(batch)
        
        teacher_logits = teacher_outputs_raw.get("ranking_scores")
        teacher_embeddings = teacher_outputs_raw.get("embeddings")
        teacher_candidates = teacher_outputs_raw.get("candidates") # (batch_size, candidate_topk)
        teacher_confidence = teacher_outputs_raw.get("confidence") # (batch_size, candidate_topk)

        # 2. 生徒モデルの出力を取得
        student_embeddings = self.student_model_module(item_seq, item_seq_len, teacher_embeddings=teacher_embeddings)
        student_logits = self.student_model_module.predict(item_seq, item_seq_len)

        # 3. 蒸留サンプルを選択
        distill_mask = self.selection_policy.select(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            ground_truth=next_item
        )

        # 4. 損失の計算
        total_loss = 0
        
        # 4.1. ランキング蒸留損失
        if self.ranking_loss_weight > 0 and distill_mask.any():
            # 重みの計算
            # weight_rank
            _lambda = 1
            _K = self.candidate_topk
            weight_static = torch.arange(1, _K + 1, dtype=torch.float32, device=self.device)
            weight_static = torch.exp(-weight_static / _lambda) # 1/exp(r)
            weight_static = torch.unsqueeze(weight_static, 0) # [1, k]
            weight_static = weight_static.repeat(student_logits.size(0), 1)
            weight_rank = weight_static / torch.sum(weight_static, dim=1, keepdim=True)
            
            # weight_com
            cf_rank_top = (-student_logits).argsort(dim=1)[:, :_K]
            common_tensor = torch.zeros_like(teacher_candidates, device=self.device)
            common_mask = teacher_candidates.unsqueeze(2) == cf_rank_top.unsqueeze(1)
            common_tensor = common_mask.any(dim=2).int() + 1e-8
            weight_com = common_tensor.to(self.device)

            # weight_confidence
            weight_confidence = torch.exp(-teacher_confidence) + 1e-8
            weight_confidence = weight_confidence / torch.sum(weight_confidence, dim=1, keepdim=True)

            # weight_fin
            weight_fin = self.gamma_position * weight_rank + self.gamma_confidence * weight_confidence + self.gamma_consistency * weight_com
            weights = weight_fin / torch.sum(weight_fin, dim=1, keepdim=True)

            ranking_kd_loss = self.ranking_kd_loss_fn(
                student_logits[distill_mask],
                teacher_candidates[distill_mask],
                weights[distill_mask]
            )
            total_loss += self.ranking_loss_weight * ranking_kd_loss
            self.log('train_ranking_kd_loss', ranking_kd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 4.2. 埋め込み蒸留損失
        if self.embedding_loss_weight > 0 and distill_mask.any():
            embedding_kd_loss = self.embedding_kd_loss_fn(
                student_embeddings[distill_mask],
                teacher_embeddings[distill_mask]
            )
            total_loss += self.embedding_loss_weight * embedding_kd_loss
            self.log('train_embedding_kd_loss', embedding_kd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 4.3. 元のタスク損失 (Cross-Entropy)
        if self.ce_loss_weight > 0:
            ce_loss = self.ce_loss_fn(student_logits, next_item)
            total_loss += self.ce_loss_weight * ce_loss
            self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        loss = self.ce_loss_fn(logits, next_item)
        
        # メトリクスの計算
        # logits (batch_size, num_items) -> predictions (List[List[int]])
        # next_item (batch_size,) -> ground_truths (List[List[int]])
        
        # predictions: 各サンプルのトップKアイテムIDを取得
        _, top_k_predictions = torch.topk(logits, k=self.metrics_k, dim=1)
        predictions_list = top_k_predictions.tolist()

        # ground_truths: 各サンプルの正解アイテムをリストのリストに変換
        ground_truths_list = [[item_id.item()] for item_id in next_item]

        metrics = calculate_metrics(predictions_list, ground_truths_list, k=self.metrics_k)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        item_seq = batch["item_seq"]
        item_seq_len = batch["item_seq_len"]
        next_item = batch["next_item"]

        logits = self.forward(item_seq, item_seq_len)
        
        # メトリクスの計算
        # predictions: 各サンプルのトップKアイテムIDを取得
        _, top_k_predictions = torch.topk(logits, k=self.metrics_k, dim=1)
        predictions_list = top_k_predictions.tolist()

        # ground_truths: 各サンプルの正解アイテムをリストのリストに変換
        ground_truths_list = [[item_id.item()] for item_id in next_item]

        metrics = calculate_metrics(predictions_list, ground_truths_list, k=self.metrics_k)
        
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.student_model_module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
