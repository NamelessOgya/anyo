from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info

from src.student.datamodule import SASRecDataModule
from src.student.models import SASRec
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer # 追加
from src.distill.trainer_distill import DistillationTrainer
from src.distill.selection_policy import AllSamplesPolicy # 現状はこれのみ
from src.student.evaluator import SASRecEvaluator # 生徒モデルの評価器

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def run_distill(cfg: DictConfig):
    # 1. ロギング、シード、Git情報の初期化
    output_dir = get_project_root() / "result" / cfg.run.dir.split('/')[-1]
    setup_logging(log_dir=output_dir / "logs")
    set_seed(cfg.seed)
    git_info = get_git_info()
    logger.info(f"Git Info: {git_info}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # 2. SASRecDataModuleのインスタンス化とデータ準備
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows
    )
    dm.prepare_data()
    dm.setup()

    # 3. 教師モデルのインスタンス化と学習済み重みのロード
    # 教師モデルは学習済みチェックポイントからロードすることを想定
    if cfg.distill.teacher_checkpoint_path and cfg.distill.teacher_checkpoint_path != "???":
        logger.info(f"Loading pre-trained teacher model from {cfg.distill.teacher_checkpoint_path}")
        # iLoRATrainerのチェックポイントをロード
        # ロード時に必要な引数を渡す
        loaded_teacher_trainer = iLoRATrainer.load_from_checkpoint(
            checkpoint_path=cfg.distill.teacher_checkpoint_path,
            ilora_model=create_teacher_model(
                cfg,
                num_items=dm.num_items,
                max_seq_len=cfg.student.max_seq_len,
                item_id_to_name=dm.item_id_to_name,
                padding_item_id=dm.padding_item_id
            ),
            num_items=dm.num_items,
            learning_rate=cfg.train.learning_rate, # ダミー値、実際には使われない
            weight_decay=cfg.train.weight_decay,   # ダミー値、実際には使われない
            metrics_k=cfg.eval.metrics_k,
            item_id_to_name=dm.item_id_to_name,
            strict=False
        )
        teacher_model_instance = loaded_teacher_trainer.model
        # 教師モデルのパラメータはフリーズする
        for param in teacher_model_instance.parameters():
            param.requires_grad = False
        teacher_model_instance.eval() # 評価モードに設定
    else:
        logger.warning("No teacher_checkpoint_path provided or it's '???'. Using randomly initialized teacher model.")
        teacher_model_instance = create_teacher_model(
            cfg,
            num_items=dm.num_items,
            max_seq_len=cfg.student.max_seq_len,
            item_id_to_name=dm.item_id_to_name,
            padding_item_id=dm.padding_item_id
        )

    # 4. 生徒モデル (SASRec) をインスタンス化
    student_model_instance = SASRec(
        num_users=dm.num_items + 1, # SASRecではユーザー数は直接使われないが、一応渡す
        num_items=dm.num_items,
        hidden_size=cfg.student.hidden_size,
        num_heads=cfg.student.num_heads,
        num_layers=cfg.student.num_layers,
        dropout_rate=cfg.student.dropout_rate,
        max_seq_len=cfg.student.max_seq_len,
        teacher_embedding_dim=teacher_model_instance.llm.config.hidden_size # 修正
    )

    # 5. DistillationTrainerのインスタンス化
    distill_trainer = DistillationTrainer(
        student_model=student_model_instance,
        teacher_model=teacher_model_instance,
        num_items=dm.num_items,
        ranking_loss_weight=cfg.distill.ranking_loss_weight,
        embedding_loss_weight=cfg.distill.embedding_loss_weight,
        ce_loss_weight=cfg.distill.ce_loss_weight,
        ranking_temperature=cfg.distill.ranking_temperature,
        embedding_loss_type=cfg.distill.embedding_loss_type,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k,
        selection_policy=AllSamplesPolicy() # TODO: cfgからポリシーを生成
    )

    # 6. PyTorch Lightning Trainerのインスタンス化と学習の実行
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best_distilled_student_model",
        monitor=f"val_recall@{cfg.eval.metrics_k}",
        mode="max",
        save_top_k=1,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard_logs")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=tb_logger,
        enable_checkpointing=True,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps
    )

    logger.info("Starting distillation training...")
    trainer.fit(distill_trainer, dm.train_dataloader(), dm.val_dataloader())
    logger.info("Distillation training finished.")

    # 7. 学習済み生徒モデルの保存
    final_model_path = output_dir / "final_distilled_student_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final distilled student model saved to {final_model_path}")

    # 8. 評価の実行
    logger.info("Starting distilled student model evaluation on test set...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best distilled student model from {best_model_path} for evaluation.")
        # DistillationTrainerからSASRecモデルを抽出して評価器に渡す
        loaded_distill_trainer = DistillationTrainer.load_from_checkpoint(
            best_model_path,
            student_model=SASRec( # student_modelは再構築が必要
                num_users=dm.num_items + 1,
                num_items=dm.num_items,
                hidden_size=cfg.student.hidden_size,
                num_heads=cfg.student.num_heads,
                num_layers=cfg.student.num_layers,
                dropout_rate=cfg.student.dropout_rate,
                max_seq_len=cfg.student.max_seq_len,
                teacher_embedding_dim=loaded_teacher_trainer.model.llm.config.hidden_size # 修正
            ),
            teacher_model=create_teacher_model( # teacher_modelも再構築が必要
                cfg,
                num_items=dm.num_items,
                max_seq_len=cfg.student.max_seq_len,
                item_id_to_name=dm.item_id_to_name,
                padding_item_id=dm.padding_item_id
            ),
            num_items=dm.num_items,
            ranking_loss_weight=cfg.distill.ranking_loss_weight,
            embedding_loss_weight=cfg.distill.embedding_loss_weight,
            ce_loss_weight=cfg.distill.ce_loss_weight,
            ranking_temperature=cfg.distill.ranking_temperature,
            embedding_loss_type=cfg.distill.embedding_loss_type,
            learning_rate=cfg.train.learning_rate, # ダミー値
            weight_decay=cfg.train.weight_decay, # ダミー値
            metrics_k=cfg.eval.metrics_k,
            selection_policy=None # ロード時は不要
        )
        # 評価器には生徒モデルのみを渡す
        evaluator = SASRecEvaluator(loaded_distill_trainer.student_model, dm, metrics_k=cfg.eval.metrics_k)
    else:
        logger.warning("No best distilled student model checkpoint found. Using final model for evaluation.")
        evaluator = SASRecEvaluator(distill_trainer.student_model, dm, metrics_k=cfg.eval.metrics_k)

    test_metrics = evaluator.evaluate(dm.test_dataloader())
    logger.info(f"Test Metrics: {test_metrics}")

    # 結果を保存
    with open(output_dir / "test_metrics.txt", "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Test metrics saved to {output_dir / 'test_metrics.txt'}")

if __name__ == "__main__":
    run_distill()
