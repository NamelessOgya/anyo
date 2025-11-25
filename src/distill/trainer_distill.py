import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader # Import DataLoader

from src.student.models import SASRec
from src.student.datamodule import SASRecDataModule
from src.teacher.interfaces import TeacherModel
from src.distill.kd_losses import RankingDistillationLoss, EmbeddingDistillationLoss, WeightedBCELoss, DROLoss, PropensityScoreCalculator
from src.distill.selection_policy import SelectionPolicy
from src.core.metrics import calculate_metrics

class DistillationTrainer(pl.LightningModule):
    def __init__(self,
                 student_model: SASRec,
                 teacher_model: TeacherModel,
                 datamodule: SASRecDataModule,
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
                 ed_weight: float,
                 num_neg_samples: int, # Add num_neg_samples parameter
                 alpha: float = 0.0, # DRO loss weight
                 beta: float = 1.0,  # DRO robust radius
                 propensity_scores: Optional[torch.Tensor] = None, # Pre-calculated ps
                 lam: float = 1.0, # Weight for importance-aware ranking distillation
                 teacher_output_dataloader: Optional[DataLoader] = None # Pre-generated teacher outputs dataloader
                 ):
        super().__init__()
        self.student_model_module = student_model # Registered as a submodule
        self.teacher_model = teacher_model
        self.datamodule = datamodule
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
        self.alpha = alpha
        self.beta = beta
        self.propensity_scores = propensity_scores
        self.lam = lam
        self.num_neg_samples = num_neg_samples # Store num_neg_samples
        self.teacher_output_dataloader = teacher_output_dataloader # Store pre-generated teacher outputs dataloader
        self.teacher_output_iterator = None # Will be initialized in on_train_epoch_start
        self.student_model_module.train() # Explicitly set to train mode here

        # 損失関数
        self.ranking_kd_loss_fn = WeightedBCELoss(alpha=self.alpha, ps=self.propensity_scores, beta=self.beta)
        self.embedding_kd_loss_fn = EmbeddingDistillationLoss(loss_type=embedding_loss_type)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        if self.alpha > 0:
            if self.propensity_scores is None:
                raise ValueError("Propensity scores must be provided if alpha > 0 for DROLoss.")
            self.dro_loss_fn = DROLoss(ps=self.propensity_scores, beta=self.beta)
        else:
            self.dro_loss_fn = None

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

    def on_train_epoch_start(self):
        """
        Called at the beginning of each training epoch.
        Initializes the iterator for teacher outputs.
        """
        if self.teacher_output_dataloader is not None:
            self.teacher_output_iterator = iter(self.teacher_output_dataloader)
        self.student_model_module.train() # Ensure student model is in train mode

    def forward(self, item_seq, item_seq_len):
        return self.student_model_module.predict(item_seq, item_seq_len)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.student_model_module.train()
        item_seq = batch["seq"]
        item_seq_len = batch["len_seq"]
        next_item_original = batch["next_item"]
        
        # 1-based ID -> 0-based index
        next_item = next_item_original.squeeze(-1) - 1

        # Debug print removed for cleaner output
        # print(f"Debug: Batch size from dm.train_dataloader(): {item_seq.size(0)}")

        # 1. 教師モデルからソフトターゲットと埋め込みを取得
        if self.teacher_output_dataloader is not None:
            # 事前生成された教師出力を利用
            teacher_outputs_for_current_batch = next(self.teacher_output_iterator)
            
            teacher_logits = teacher_outputs_for_current_batch["ranking_scores"].to(self.device)
            teacher_embeddings = teacher_outputs_for_current_batch["embeddings"].to(self.device)
            teacher_candidates = teacher_outputs_for_current_batch["candidates"].to(self.device)
            teacher_confidence = teacher_outputs_for_current_batch["confidence"].to(self.device)
            
            # print(f"Debug: Batch size from teacher_output_dataloader(): {teacher_embeddings.size(0)}")
            
            teacher_outputs_raw = {
                "ranking_scores": teacher_logits,
                "embeddings": teacher_embeddings,
                "candidates": teacher_candidates,
                "confidence": teacher_confidence,
            }
        else:
            # オンザフライで教師出力を生成
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
            ground_truth=next_item_original.squeeze(-1) # SelectionPolicy might expect 1-based? Let's check.
            # GroundTruthErrorPolicy uses gather with ground_truth. 
            # If student_logits is 0-based (size C), ground_truth must be 0-based.
            # So we should pass next_item (0-based).
        )
        # Wait, let's re-check SelectionPolicy.
        # GroundTruthErrorPolicy: ground_truth_logits = student_logits.gather(1, ground_truth.unsqueeze(1))
        # If student_logits is (B, C), ground_truth must be in [0, C-1].
        # So we MUST pass 0-based index.
        distill_mask = self.selection_policy.select(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            ground_truth=next_item
        )

        # 4. ネガティブサンプリング (DLLM2Rec main.pyから移植)
        real_batch_size = item_seq.size(0)
        num_items_total = self.num_items
        
        zeros_tensor = torch.zeros((real_batch_size, num_items_total + 1), device=self.device)
        
        clamped_item_seq = torch.clamp(item_seq, max=num_items_total)
        zeros_tensor.scatter_(1, clamped_item_seq, 1)
        
        zeros_tensor.scatter_(1, next_item_original.unsqueeze(1), 1) # next_item_original is 1-based, matches num_items_total+1 size
        
        zeros_tensor = zeros_tensor[:, :num_items_total]
        
        neg_tensor = 1 - zeros_tensor
        
        if self.num_neg_samples > 0:
            neg_samples = torch.multinomial(
                neg_tensor, self.num_neg_samples, replacement=True
            )
        else:
            neg_samples = torch.empty((real_batch_size, 0), dtype=torch.long, device=self.device)

        # 5. 損失の計算
        total_loss = 0
        
        # 5.1. ランキング蒸留損失
        if self.ranking_loss_weight > 0 and distill_mask.any():
            _lambda = 1
            _K = self.candidate_topk
            weight_static = torch.arange(1, _K + 1, dtype=torch.float32, device=self.device)
            weight_static = torch.exp(-weight_static / _lambda)
            weight_static = torch.unsqueeze(weight_static, 0).repeat(student_logits.size(0), 1)
            weight_rank = weight_static / torch.sum(weight_static, dim=1, keepdim=True)
            
            cf_rank_top = (-student_logits).argsort(dim=1)[:, :_K]
            common_tensor = (teacher_candidates.unsqueeze(2) == cf_rank_top.unsqueeze(1)).any(dim=2).int() + 1e-8
            weight_com = common_tensor.to(self.device)

            weight_confidence = torch.exp(-teacher_confidence) + 1e-8
            weight_confidence = weight_confidence / torch.sum(weight_confidence, dim=1, keepdim=True)

            weight_fin = self.gamma_position * weight_rank + self.gamma_confidence * weight_confidence + self.gamma_consistency * weight_com
            weights = weight_fin / torch.sum(weight_fin, dim=1, keepdim=True)

            ranking_kd_loss = self.ranking_kd_loss_fn(
                student_logits[distill_mask],
                teacher_candidates[distill_mask],
                weights[distill_mask],
                neg_samples[distill_mask]
            )
            total_loss += self.lam * self.ranking_loss_weight * ranking_kd_loss
            self.log('train_ranking_kd_loss', ranking_kd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 5.2. 埋め込み蒸留損失
        if self.embedding_loss_weight > 0 and distill_mask.any():
            embedding_kd_loss = self.embedding_kd_loss_fn(
                student_embeddings[distill_mask],
                teacher_embeddings[distill_mask]
            )
            total_loss += self.embedding_loss_weight * embedding_kd_loss
            self.log('train_embedding_kd_loss', embedding_kd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 5.3. 元のタスク損失 (Cross-Entropy + DRO)
        if self.ce_loss_weight > 0:
            ce_loss = self.ce_loss_fn(student_logits, next_item)
            if self.alpha > 0 and self.dro_loss_fn is not None:
                # Pass 0-based target to DROLoss
                dro_loss = self.dro_loss_fn(student_logits, next_item)
                ce_loss = ce_loss + self.alpha * dro_loss
                self.log('train_ce_dro_loss', dro_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            total_loss += self.ce_loss_weight * ce_loss
            self.log('train_ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        item_seq = batch["seq"]
        item_seq_len = batch["len_seq"]
        next_item_original = batch["next_item"]
        
        # 1-based ID -> 0-based index
        next_item = next_item_original.squeeze(-1) - 1

        logits = self.forward(item_seq, item_seq_len)
        loss = self.ce_loss_fn(logits, next_item)
        
        # メトリクスの計算
        # logits (batch_size, num_items) -> predictions (List[List[int]])
        # next_item (batch_size,) -> ground_truths (List[List[int]])
        
        # predictions: 各サンプルのトップKアイテムIDを取得 (0-based indices)
        _, top_k_predictions = torch.topk(logits, k=self.metrics_k, dim=1)
        predictions_list = top_k_predictions.tolist()

        # ground_truths: 各サンプルの正解アイテムをリストのリストに変換
        # padding_item_idを除外
        ground_truths_list = []
        for item_id_tensor in next_item_original:
            item_id = item_id_tensor.item()
            if item_id != self.datamodule.padding_item_id:
                # Convert 1-based ID to 0-based index to match predictions
                ground_truths_list.append([item_id - 1])
        
        metrics = calculate_metrics(predictions_list, ground_truths_list, k=self.metrics_k)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        item_seq = batch["seq"]
        item_seq_len = batch["len_seq"]
        next_item_original = batch["next_item"]
        
        # 1-based ID -> 0-based index
        next_item = next_item_original.squeeze(-1) - 1

        logits = self.forward(item_seq, item_seq_len)
        loss = self.ce_loss_fn(logits, next_item)
        
        # メトリクスの計算
        _, top_k_predictions = torch.topk(logits, k=self.metrics_k, dim=1)
        predictions_list = top_k_predictions.tolist()

        ground_truths_list = []
        for item_id_tensor in next_item_original:
            item_id = item_id_tensor.item()
            if item_id != self.datamodule.padding_item_id:
                # Convert 1-based ID to 0-based index to match predictions
                ground_truths_list.append([item_id - 1])
        
        metrics = calculate_metrics(predictions_list, ground_truths_list, k=self.metrics_k)
        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.student_model_module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
