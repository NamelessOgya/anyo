import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl
import torch
from typing import Dict, Any
from pathlib import Path

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info

from src.student.datamodule import SASRecDataModule
from src.student.models import SASRec
from src.student.trainer_baseline import SASRecTrainer
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from src.distill.trainer_distill import DistillationTrainer
from src.student.evaluator import SASRecEvaluator

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def run_eval_all(cfg: DictConfig):
    # 1. ロギング、シード、Git情報の初期化
    output_dir = Path(cfg.run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir / "logs")
    set_seed(cfg.seed)
    git_info = get_git_info()
    logger.info(f"Git Info: {git_info}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # 2. SASRecDataModuleのインスタンス化とデータ準備
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.eval.batch_size, # 評価用のバッチサイズ
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers # 評価時もnum_workersを使用
    )
    dm.prepare_data()
    dm.setup()

    all_results: Dict[str, Dict[str, float]] = {}

    # 3. ベースライン生徒モデルの評価
    if cfg.eval.baseline_student_checkpoint_path:
        logger.info("Evaluating Baseline Student Model...")
        try:
            baseline_student_trainer = SASRecTrainer.load_from_checkpoint(
                cfg.eval.baseline_student_checkpoint_path,
                num_users=dm.num_items + 1,
                num_items=dm.num_items,
                hidden_size=cfg.student.hidden_size,
                num_heads=cfg.student.num_heads,
                num_layers=cfg.student.num_layers,
                dropout_rate=cfg.student.dropout_rate,
                max_seq_len=cfg.student.max_seq_len,
                learning_rate=cfg.train.learning_rate, # ダミー値
                weight_decay=cfg.train.weight_decay, # ダミー値
                metrics_k=cfg.eval.metrics_k
            )
            evaluator = SASRecEvaluator(baseline_student_trainer, dm, metrics_k=cfg.eval.metrics_k)
            baseline_metrics = evaluator.evaluate(dm.test_dataloader())
            all_results["baseline_student"] = baseline_metrics
            logger.info(f"Baseline Student Metrics: {baseline_metrics}")
        except Exception as e:
            logger.error(f"Failed to evaluate Baseline Student Model: {e}")

    # 4. 蒸留済み生徒モデルの評価
    if cfg.eval.distilled_student_checkpoint_path:
        logger.info("Evaluating Distilled Student Model...")
        try:
            # DistillationTrainerをロードし、そこからstudent_modelを抽出
            # 教師モデルはダミーで再構築が必要
            dummy_teacher_model_instance = create_teacher_model(
                cfg,
                num_items=dm.num_items,
                max_seq_len=cfg.student.max_seq_len,
                item_id_to_name=dm.item_id_to_name,
                padding_item_id=dm.padding_item_id
            )
            distilled_trainer = DistillationTrainer.load_from_checkpoint(
                cfg.eval.distilled_student_checkpoint_path,
                student_model=SASRec( # student_modelは再構築が必要
                    num_users=dm.num_items + 1,
                    num_items=dm.num_items,
                    hidden_size=cfg.student.hidden_size,
                    num_heads=cfg.student.num_heads,
                    num_layers=cfg.student.num_layers,
                    dropout_rate=cfg.student.dropout_rate,
                    max_seq_len=cfg.student.max_seq_len
                ),
                teacher_model=dummy_teacher_model_instance, # ダミーの教師モデルインスタンス
                num_items=dm.num_items,
                ranking_loss_weight=cfg.distill.ranking_loss_weight, # ダミー値
                embedding_loss_weight=cfg.distill.embedding_loss_weight, # ダミー値
                ce_loss_weight=cfg.distill.ce_loss_weight, # ダミー値
                ranking_temperature=cfg.distill.ranking_temperature, # ダミー値
                embedding_loss_type=cfg.distill.embedding_loss_type, # ダミー値
                learning_rate=cfg.train.learning_rate, # ダミー値
                weight_decay=cfg.train.weight_decay, # ダミー値
                metrics_k=cfg.eval.metrics_k,
                selection_policy=None, # ロード時は不要
                strict=False # ここにstrict=Falseを追加
            )
            evaluator = SASRecEvaluator(distilled_trainer.student_model, dm, metrics_k=cfg.eval.metrics_k)
            distilled_metrics = evaluator.evaluate(dm.test_dataloader())
            all_results["distilled_student"] = distilled_metrics
            logger.info(f"Distilled Student Metrics: {distilled_metrics}")
        except Exception as e:
            logger.error(f"Failed to evaluate Distilled Student Model: {e}")

    # 5. 教師モデルの評価
    if cfg.eval.teacher_checkpoint_path:
        logger.info("Evaluating Teacher Model...")
        try:
            # iLoRATrainerをロードし、そこからiLoRAModelを抽出
            teacher_model_instance = create_teacher_model(
                cfg,
                num_items=dm.num_items,
                max_seq_len=cfg.student.max_seq_len,
                item_id_to_name=dm.item_id_to_name,
                padding_item_id=dm.padding_item_id
            )
            teacher_trainer = iLoRATrainer.load_from_checkpoint(
                cfg.eval.teacher_checkpoint_path,
                ilora_model=teacher_model_instance, # iLoRAModelインスタンスを渡す必要がある
                num_items=dm.num_items,
                learning_rate=cfg.train.learning_rate, # ダミー値
                weight_decay=cfg.train.weight_decay, # ダミー値
                metrics_k=cfg.eval.metrics_k,
                item_id_to_name=dm.item_id_to_name,
                # iLoRAModelのハイパーパラメータを明示的に渡す
                llm_model_name=cfg.teacher.llm_model_name,
                num_lora_experts=cfg.teacher.num_lora_experts,
                lora_r=cfg.teacher.lora_r,
                lora_alpha=cfg.teacher.lora_alpha,
                lora_dropout=cfg.teacher.lora_dropout,
                hidden_size=cfg.teacher.hidden_size,
                dropout_rate=cfg.teacher.dropout_rate,
                max_seq_len=cfg.student.max_seq_len, # max_seq_lenはstudentから取得
                padding_item_id=dm.padding_item_id
            )
            # 教師モデルの評価は、iLoRATrainerのtest_stepを直接呼び出す
            # SASRecEvaluatorはSASRecモデルを想定しているため、直接は使えない
            # ここでは、Trainerを使ってtest_stepを実行
            temp_trainer = pl.Trainer(
                accelerator=cfg.train.accelerator,
                devices=cfg.train.devices,
                logger=False,
                enable_checkpointing=False
            )
            teacher_test_results = temp_trainer.test(teacher_trainer, dm.test_dataloader())
            # test_stepのログからメトリクスを抽出
            teacher_metrics = {k: v for k, v in teacher_test_results[0].items() if "test_" in k}
            all_results["teacher_model"] = teacher_metrics
            logger.info(f"Teacher Model Metrics: {teacher_metrics}")
        except Exception as e:
            logger.error(f"Failed to evaluate Teacher Model: {e}")

    # 6. すべての評価結果をまとめて出力
    logger.info("\n--- All Evaluation Results ---")
    for model_name, metrics in all_results.items():
        logger.info(f"Model: {model_name}")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # 結果をJSONファイルとして保存
    import json
    with open(output_dir / "all_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"All evaluation results saved to {output_dir / 'all_evaluation_results.json'}")

if __name__ == "__main__":
    run_eval_all()
