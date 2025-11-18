import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from src.student.models import SASRec
from src.student.trainer_baseline import SASRecTrainer
from src.student.datamodule import SASRecDataModule
from src.core.metrics import calculate_metrics

class SASRecEvaluator:
    def __init__(self, 
                 model: SASRec, # SASRecTrainerからSASRecに変更
                 datamodule: SASRecDataModule, 
                 metrics_k: int = 10):
        self.model = model
        self.datamodule = datamodule
        self.metrics_k = metrics_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # 評価モードに設定

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        all_predictions: List[List[int]] = []
        all_ground_truths: List[List[int]] = []

        for batch in dataloader:
            seq = batch["seq"].to(self.device)
            len_seq = batch["len_seq"].to(self.device)
            next_item = batch["next_item"].to(self.device)

            logits = self.model.predict(seq, len_seq)
            
            # デバッグログを追加
            if not hasattr(self, '_debug_logged'): # 最初のバッチのみログを出力
                print(f"Debug: Logits shape: {logits.shape}")
                print(f"Debug: Logits (first 5 values of first sample): {logits[0, :5]}")

            # トップKの予測アイテムIDを取得
            _, predicted_indices = torch.topk(logits, self.metrics_k, dim=-1)

            if not hasattr(self, '_debug_logged'): # 最初のバッチのみログを出力
                print(f"Debug: Predicted indices shape: {predicted_indices.shape}")
                print(f"Debug: Predicted indices (first sample): {predicted_indices[0]}")
                self._debug_logged = True # ログ出力済みフラグを設定
            
            all_predictions.extend(predicted_indices.tolist())
            all_ground_truths.extend([[item.item()] for item in next_item])

        # 全ての予測と正解に基づいてメトリクスを計算
        # padding_item_idを除外
        filtered_ground_truths = []
        for gt_list in all_ground_truths:
            filtered_gt = [item for item in gt_list if item != self.datamodule.padding_item_id]
            if filtered_gt: # 空リストにならないように
                filtered_ground_truths.append(filtered_gt)
            else:
                filtered_ground_truths.append([]) # 空の場合は空リストを保持
        
        overall_metrics = calculate_metrics(all_predictions, filtered_ground_truths, self.metrics_k)
        return overall_metrics

if __name__ == "__main__":
    # テスト用のダミーデータとデータモジュール
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor

    # ダミーデータモジュール
    dm = SASRecDataModule(batch_size=4, max_seq_len=50, num_workers=0) # num_workers=0 for Windows/debugging
    dm.prepare_data()
    dm.setup()

    # トレーナーのインスタンス化
    num_users_dummy = 1000 
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

    # ダミーの学習済みモデルをロードする代わりに、ここでは初期化されたモデルを使用
    # 実際の使用では、trainer_model = SASRecTrainer.load_from_checkpoint("path/to/checkpoint.ckpt") のようにロードする

    # Evaluatorのインスタンス化
    evaluator = SASRecEvaluator(trainer_model, dm, metrics_k=10)

    print("\n--- Starting dummy evaluation on validation set ---")
    val_metrics = evaluator.evaluate(dm.val_dataloader())
    print(f"Validation Metrics: {val_metrics}")

    print("\n--- Starting dummy evaluation on test set ---")
    test_metrics = evaluator.evaluate(dm.test_dataloader())
    print(f"Test Metrics: {test_metrics}")

    print("\nSASRecEvaluator test finished!")
