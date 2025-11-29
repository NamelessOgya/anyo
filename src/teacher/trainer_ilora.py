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
        distill_lambda: float = 0.0,
        distill_loss_type: str = "mse",
        distill_decay_type: str = "none",
        distill_min_lambda: float = 0.0,
        distill_decay_steps: Optional[int] = None,
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
        self.distill_lambda = distill_lambda
        self.distill_loss_type = distill_loss_type
        self.distill_decay_type = distill_decay_type
        self.distill_min_lambda = distill_min_lambda
        self.distill_decay_steps = distill_decay_steps

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        学習ステップのフォワードパス。
        """
        return self.model(batch)

    def _get_current_lambda(self) -> float:
        """
        現在の学習ステップに基づいて、蒸留損失の重み（Lambda）を計算します。
        設定された減衰スケジュール（linear, cosine, exponential）に従って値を変化させます。
        """
        if self.distill_lambda <= 0 or self.distill_decay_type == "none":
            return self.distill_lambda
            
        # 減衰にかける期間（ステップ数）を決定
        if self.distill_decay_steps is not None and self.distill_decay_steps > 0:
            decay_duration = self.distill_decay_steps
        elif self.trainer.max_steps and self.trainer.max_steps > 0:
            decay_duration = self.trainer.max_steps
        else:
            # フォールバック: max_epochsから推定するか、初期値を返す
            return self.distill_lambda

        current_step = self.global_step
        progress = min(current_step / decay_duration, 1.0)
        
        start = self.distill_lambda
        end = self.distill_min_lambda
        
        if self.distill_decay_type == "linear":
            # 線形減衰
            return start - (start - end) * progress
        elif self.distill_decay_type == "cosine":
            # コサイン減衰
            import math
            return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))
        elif self.distill_decay_type == "exponential":
            # 指数関数的減衰: start * (end/start)^progress
            if start == 0: return 0.0
            if end <= 0: end = 1e-6 
            return start * ((end / start) ** progress)
        else:
            return start

    def _compute_distill_loss(self, current_embeddings: torch.Tensor, original_embeddings: torch.Tensor) -> torch.Tensor:
        """
        設定されたタイプに基づいて蒸留損失（Reverse Distillation Loss）を計算します。
        
        Args:
            current_embeddings: 現在のStudentモデル（学習中）のアイテム埋め込み
            original_embeddings: 教師（初期状態のStudent）のアイテム埋め込み
        """
        if self.distill_loss_type == "mse":
            # 平均二乗誤差: ベクトルの大きさと方向の両方を近づける
            return F.mse_loss(current_embeddings, original_embeddings)
        elif self.distill_loss_type == "l1":
            # L1損失（平均絶対誤差）: 外れ値に対してロバスト
            return F.l1_loss(current_embeddings, original_embeddings)
        elif self.distill_loss_type == "huber":
            # Huber損失: MSEとL1のハイブリッド
            return F.huber_loss(current_embeddings, original_embeddings)
        elif self.distill_loss_type == "cosine":
            # コサイン埋め込み損失: 1 - cosine_similarity
            # ベクトルの方向のみを近づけ、大きさは無視する
            # F.cosine_embedding_loss は input1, input2, target を期待するが、
            # ここでは単純に平均コサイン類似度を計算して 1 から引く
            cos_sim = F.cosine_similarity(current_embeddings, original_embeddings, dim=-1)
            return 1.0 - cos_sim.mean()
        elif self.distill_loss_type == "contrastive":
            # 単純な対照学習損失 (InfoNCE形式)
            # 正例ペア: (current[i], original[i]) -> 類似度を上げる
            # 負例ペア: (current[i], original[j]) where j != i -> 類似度を下げる
            
            # 安定性のために正規化
            curr_norm = F.normalize(current_embeddings, p=2, dim=-1)
            orig_norm = F.normalize(original_embeddings, p=2, dim=-1)
            
            # 類似度行列 (N, N)
            logits = torch.matmul(curr_norm, orig_norm.T)
            
            # 温度パラメータ (通常 0.1 程度)
            temperature = 0.1
            logits = logits / temperature
            
            # 正解ラベルは対角成分 (0, 1, 2, ... N-1)
            labels = torch.arange(logits.size(0), device=logits.device)
            
            return F.cross_entropy(logits, labels)
        else:
            # 未知のタイプの場合はデフォルトでMSEを使用
            return F.mse_loss(current_embeddings, original_embeddings)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        1回の学習ステップ。
        iLoRAのランキングスコアを使用したCross Entropy Lossと、
        オプションでReverse Distillation Lossを計算します。
        """
        start_time = time.time()
        
        # LLMのフォワードパス
        outputs = self.model(batch)
        
        # Ranking Loss (Cross Entropy)
        # outputsはLLMの生出力 (Batch, Seq, Hidden)
        # Ranking Loss (Cross Entropy)
        # outputsはLLMの生出力 (Batch, Seq, Hidden)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] # (B, H)
        
        # 正解アイテムのID (1-based)
        target_items = batch["next_item"] # (batch_size,)

        if self.model.use_item_embeddings_head:
             # Sampled Softmax Logic
             num_samples = 2048
             num_total_items = self.model.rec_model.item_embeddings.num_embeddings
             
             # 1. 正解アイテムをユニークにする
             unique_targets = torch.unique(target_items)
             
             # 2. 負例をサンプリング (パディング(0)を除く 1 ~ num_total_items-1)
             # 簡易的なランダムサンプリング
             num_negatives = num_samples - len(unique_targets)
             if num_negatives > 0:
                 negative_samples = torch.randint(1, num_total_items, (num_negatives,), device=self.device)
                 sampled_indices = torch.cat([unique_targets, negative_samples])
             else:
                 sampled_indices = unique_targets
                 
             sampled_indices = torch.unique(sampled_indices) # 重複排除
             
             # 3. Embedding計算 (サンプリングされたアイテムのみ)
             sampled_item_embs_rec = self.model.rec_model.item_embeddings(sampled_indices)
             sampled_item_embs_llm = self.model.projector(sampled_item_embs_rec.to(self.model.projector.model[0].weight.dtype))
             
             # 4. ロジット計算 (B, Num_Sampled)
             logits = last_hidden_state @ sampled_item_embs_llm.T
             
             # 5. ターゲットの再マッピング
             # target_items が sampled_indices のどこにあるかを探す
             # searchsortedを使うためにソートが必要だが、ここでは簡易的にbroadcastで比較
             # (B, 1) == (Num_Sampled,) -> (B, Num_Sampled) -> nonzero
             # メモリ効率のため、各ターゲットについてインデックスを取得
             
             # map_indices: target_itemsの値をsampled_indices内のインデックスに変換
             # sampled_indicesは小さいので、辞書や検索で対応可能
             # GPU上での効率的な実装:
             # target_items (B)
             # sampled_indices (S)
             # index_mapping tableを作成するのはSが大きいと大変だが、S=2048ならOK
             
             # ターゲットIDをインデックスに変換
             # sampled_indices を値 -> インデックス のマップにするのは難しい (GPU tensor)
             # しかし、target_itemsは必ずsampled_indicesに含まれている
             
             # 方法: target_items (B, 1) と sampled_indices (1, S) を比較
             # mask = (target_items.unsqueeze(1) == sampled_indices.unsqueeze(0)) # (B, S)
             # new_targets = mask.nonzero()[:, 1] # (B,)
             
             # これでOK
             mask = (target_items.unsqueeze(1) == sampled_indices.unsqueeze(0))
             new_targets = torch.argmax(mask.float(), dim=1) # 最初にマッチしたインデックス
             
             loss = F.cross_entropy(logits, new_targets)
             
        else:
             # Linear Head: 線形層による予測 (全アイテム計算)
             ranking_scores = self.model.item_prediction_head(last_hidden_state)
             # ranking_scoresは0-based index (0 -> Item 1) なので、target_itemsから1を引く
             loss = F.cross_entropy(ranking_scores, target_items - 1)

        # Reverse Distillation Loss (Embedding Imitation)
        current_lambda = self._get_current_lambda()
        
        if current_lambda > 0 and hasattr(self.model, "student_item_embeddings"):
            current_embeddings = self.model.rec_model.item_embeddings.weight
            original_embeddings = self.model.student_item_embeddings
            
            # Distillation Lossの計算
            reg_loss = self._compute_distill_loss(current_embeddings, original_embeddings)
            
            loss += current_lambda * reg_loss
            
            self.log("train_reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("distill_lambda", current_lambda, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
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
        # 上位K個のアイテムを取得
        _, topk_indices = torch.topk(ranking_scores, k=self.metrics_k, dim=-1) # (batch_size, k)
        
        # topk_indicesは0-basedなので、1を足してItem IDに変換
        topk_item_ids = topk_indices + 1
        
        hits = torch.zeros(target_items.size(0), device=self.device)
        ndcgs = torch.zeros(target_items.size(0), device=self.device)

        for i in range(target_items.size(0)):
            target = target_items[i]
            if target in topk_item_ids[i]:
                hits[i] = 1.0
                # NDCG計算: 正解がランクのどこにあるか
                rank = (topk_item_ids[i] == target).nonzero(as_tuple=True)[0].item()
                ndcgs[i] = 1.0 / torch.log2(torch.tensor(rank + 2.0))

        self.log(f"val_hr@{self.metrics_k}", hits.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_ndcg@{self.metrics_k}", ndcgs.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": 0.0} # ダミーの損失

    def configure_optimizers(self):
        """
        オプティマイザとスケジューラの設定。
        """
        # 重み減衰の適用（バイアスとLayerNormを除く）
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
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
        
        # 0-based indices -> 1-based Item IDs
        predicted_item_ids = predicted_indices + 1
        predictions = predicted_item_ids.tolist()
        
        ground_truths_list = []
        for item_id_tensor in target_items:
            item_id = item_id_tensor.item()
            ground_truths_list.append([item_id])

        metrics = calculate_metrics(predictions, ground_truths_list, k=self.metrics_k)
        for metric_name, metric_value in metrics.items():
            self.log(f"test_{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_loss": 0.0} # Dummy loss



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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        チェックポイント保存時に呼び出されます。
        凍結されたパラメータ（ベースLLMの重みなど）をstate_dictから削除して、
        ファイルサイズを削減します。
        """
        state_dict = checkpoint["state_dict"]
        
        # 学習可能なパラメータの名前を取得
        trainable_param_names = {n for n, p in self.named_parameters() if p.requires_grad}
        # 全パラメータの名前を取得（バッファの判定用）
        all_param_names = {n for n, p in self.named_parameters()}
        
        keys_to_keep = []
        for key in state_dict.keys():
            if key in trainable_param_names:
                keys_to_keep.append(key)
            elif key not in all_param_names:
                # named_parametersに含まれないキーはバッファ（running_meanなど）とみなして保存
                keys_to_keep.append(key)
        
        # フィルタリングされたstate_dictを作成
        new_state_dict = {k: v for k, v in state_dict.items() if k in keys_to_keep}
        checkpoint["state_dict"] = new_state_dict
        
        logger.info(f"Checkpoint optimized: Reduced state_dict from {len(state_dict)} to {len(new_state_dict)} keys.")

if __name__ == "__main__":
    # テスト用のダミーデータとデータモジュール
    from src.student.datamodule import SASRecDataModule # データモジュールは共通
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor
    from transformers import AutoModel, AutoTokenizer # LLMとTokenizerをロードするために追加
    from src.teacher.mlp_projector import MLPProjector # Added import for MLPProjector

    # iLoRAModelのダミー設定
    ilora_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 6,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })

    # LLMとTokenizerをロード
    llm = AutoModel.from_pretrained(ilora_cfg.llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(ilora_cfg.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))

    # ダミーデータモジュール
    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="data/ml-1m",
        batch_size=4,
        max_seq_len=50,
        num_workers=0,
        limit_data_rows=100,
        tokenizer=tokenizer
    )
    dm.prepare_data()
    dm.setup()

    # ダミーのrec_modelとprojectorを作成
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size_rec, num_items_rec):
            super().__init__()
            self.hidden_size = hidden_size_rec
            self.item_embeddings = nn.Embedding(num_items_rec + 1, hidden_size_rec)
            self.cacu_x = lambda x: self.item_embeddings(x)
            self.cacul_h = lambda x, y: torch.randn(x.shape[0], hidden_size_rec).to(x.device)
        
        def get_full_sequence_representations(self, item_seq, item_seq_len):
            batch_size, seq_len = item_seq.shape
            return torch.randn(batch_size, seq_len, self.hidden_size).to(item_seq.device)

        def _get_last_item_representation(self, item_seq, item_seq_len):
            batch_size = item_seq.shape[0]
            return torch.randn(batch_size, self.hidden_size).to(item_seq.device)
    
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
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate,
        item_id_to_name=dm.mapped_id_to_title,
        padding_item_id=tokenizer.pad_token_id,
        rec_model=dummy_rec_model,
        projector=dummy_projector,
        candidate_topk=10,
        llm_dtype=llm.dtype
    )

    # iLoRATrainerのインスタンス化
    ilora_trainer = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=1e-4,
        weight_decay=0.01,
        metrics_k=10,
        item_id_to_name=dm.mapped_id_to_title
    )

    # PyTorch Lightning Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu", # テスト用にCPUを使用
        devices=1,
        logger=False, # テスト時はロガーを無効化
        enable_checkpointing=False,
    )

    print("\n--- Starting dummy iLoRA training ---")
    trainer.fit(ilora_trainer, dm.train_dataloader(), dm.val_dataloader())
    print("Dummy iLoRA training finished.")

    print("\n--- Starting dummy iLoRA testing ---")
    trainer.test(ilora_trainer, dm.test_dataloader())
    print("Dummy iLoRA testing finished.")
