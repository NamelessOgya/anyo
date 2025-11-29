from src.core.config_utils import load_hydra_config
from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info
from omegaconf import OmegaConf

from src.student.datamodule import SASRecDataModule
from src.student.models import SASRec
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from src.distill.trainer_distill import DistillationTrainer
from src.distill.selection_policy import AllSamplesPolicy
from src.student.evaluator import SASRecEvaluator
from src.distill.kd_losses import PropensityScoreCalculator
from src.distill.teacher_output_dataset import TeacherOutputDataset, teacher_output_collate_fn
from torch.utils.data import DataLoader
import torch

import logging
import sys
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

def run_experiment(cfg):
    # Use cfg.run.dir
    output_dir = Path(cfg.run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"!!! SCRIPT RUNNING. OUTPUT DIR: {output_dir} !!!")

    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    setup_logging(log_dir=output_dir / "logs")
    set_seed(cfg.seed)
    git_info = get_git_info()
    logger.info(f"Git Info: {git_info}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # 2. SASRecDataModuleの初期インスタンス化
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 3. Determine Distillation Type and Load Teacher Config
    teacher_checkpoint_file_path = Path(cfg.distill.teacher_checkpoint_path) if cfg.distill.teacher_checkpoint_path else None
    distill_type = cfg.distill.get("type", "dllm2rec") # Default fallback
    teacher_cfg = None

    if teacher_checkpoint_file_path and teacher_checkpoint_file_path.exists():
        logger.info(f"Found teacher checkpoint at: {teacher_checkpoint_file_path}")
        # Try to load config from checkpoint directory
        # checkpoint_path/checkpoints/teacher-epoch=XX.ckpt -> parent -> parent -> config.yaml
        teacher_run_dir = teacher_checkpoint_file_path.parents[1]
        teacher_cfg_path = teacher_run_dir / "config.yaml"
        
        if teacher_cfg_path.exists():
            teacher_cfg = OmegaConf.load(teacher_cfg_path)
            logger.info(f"Loaded teacher config from: {teacher_cfg_path}")
            
            # Infer type from teacher config
            if "model_type" in teacher_cfg.teacher:
                model_type = teacher_cfg.teacher.model_type
                if model_type == "bigrec":
                    distill_type = "generative"
                elif model_type == "ilora":
                    distill_type = "dllm2rec"
                logger.info(f"Inferred distillation type '{distill_type}' from teacher model_type '{model_type}'")
            else:
                logger.warning("Teacher config does not have 'model_type'. Using 'distill.type' from current config.")
        else:
            logger.warning(f"Teacher config not found at {teacher_cfg_path}. Using 'distill.type' from current config.")
    else:
        logger.info("No teacher checkpoint provided or found. Using 'distill.type' from current config.")

    logger.info(f"Final Distillation Type: {distill_type}")

    if distill_type == "generative":

        # 4. 教師モデルのインスタンス化と学習済み重みのロード (using the teacher's specific config)
        logger.info(f"Loading pre-trained teacher model from {teacher_checkpoint_file_path}")
        teacher_model_instance = create_teacher_model(
            teacher_cfg, # Use the specific config from the teacher run
            num_items=dm.num_items,
            max_seq_len=teacher_cfg.student.max_seq_len,
            item_id_to_name=dm.mapped_id_to_title,
            padding_item_id=dm.padding_item_id,
            candidate_topk=teacher_cfg.distill.candidate_topk
        )
        loaded_teacher_trainer = iLoRATrainer.load_from_checkpoint(
            checkpoint_path=teacher_checkpoint_file_path,
            ilora_model=teacher_model_instance,
            num_items=dm.num_items,
            learning_rate=teacher_cfg.train.learning_rate,
            weight_decay=teacher_cfg.train.weight_decay,
            metrics_k=teacher_cfg.eval.metrics_k,
            item_id_to_name=dm.mapped_id_to_title,
            strict=False
        )
        teacher_model_instance = loaded_teacher_trainer.model
        for param in teacher_model_instance.parameters():
            param.requires_grad = False
        teacher_model_instance.eval()

        # 5. Load pre-generated teacher outputs
        logger.info(f"Loading pre-generated teacher outputs from {teacher_outputs_batches_dir_path}")
        teacher_output_dataset = TeacherOutputDataset(teacher_outputs_batches_dir_path)
        teacher_output_dataloader = DataLoader(
            teacher_output_dataset,
            batch_size=1, # Each file is already a batch
            shuffle=False,
            num_workers=0,
            collate_fn=teacher_output_collate_fn
        )
        
        # 6. 傾向スコア (Propensity Scores) の計算
        train_next_items = []
        for batch in dm.train_dataloader():
            train_next_items.extend(batch["next_item"].squeeze(-1).tolist())
        ps_calculator = PropensityScoreCalculator(
            item_num=dm.num_items + 1,
            train_next_items=train_next_items,
            power=cfg.distill.ps_power
        )
        propensity_scores = ps_calculator.get_ps()
        logger.info(f"Propensity scores calculated. Shape: {propensity_scores.shape}")

        # 7. 生徒モデル (SASRec) をインスタンス化
        student_model_instance = SASRec(
            num_items=dm.num_items,
            hidden_size=cfg.student.hidden_size,
            num_heads=cfg.student.num_heads,
            num_layers=cfg.student.num_layers,
            dropout_rate=cfg.student.dropout_rate,
            max_seq_len=cfg.student.max_seq_len,
            teacher_embedding_dim=teacher_model_instance.llm.config.hidden_size,
            padding_item_id=dm.padding_item_id
        )

        # 8. DistillationTrainerのインスタンス化
        distill_trainer = DistillationTrainer(
            student_model=student_model_instance,
            teacher_model=teacher_model_instance,
            datamodule=dm,
            num_items=dm.num_items,
            ranking_loss_weight=cfg.distill.ranking_loss_weight,
            embedding_loss_weight=cfg.distill.embedding_loss_weight,
            ce_loss_weight=cfg.distill.ce_loss_weight,
            ranking_temperature=cfg.distill.ranking_temperature,
            embedding_loss_type=cfg.distill.embedding_loss_type,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            metrics_k=cfg.eval.metrics_k,
            selection_policy=AllSamplesPolicy(),
            gamma_position=cfg.distill.gamma_position,
            gamma_confidence=cfg.distill.gamma_confidence,
            lambda_rank=cfg.distill.ranking_loss_weight,
            num_candidates=cfg.distill.candidate_topk,
            item_id_to_name=dm.mapped_id_to_title,
            alpha=cfg.distill.alpha,
            beta=cfg.distill.beta,
            propensity_scores=propensity_scores,
            lam=cfg.distill.lam,
            num_neg_samples=cfg.distill.num_neg_samples,
            teacher_output_dataloader=teacher_output_dataloader
        )

    else:
        # dllm2rec (iLoRA)
        logger.info(f"Loading iLoRA teacher model from {teacher_checkpoint_file_path}")
        
        # Load Teacher Config if not already loaded (though we tried earlier)
        # If teacher_cfg is None, we might need to rely on current config or fail?
        # iLoRATrainer.load_from_checkpoint handles loading.
        
        # We need to instantiate the model first?
        # iLoRATrainer.load_from_checkpoint(..., ilora_model=instance)
        # We need to create the instance.
        
        # If teacher_cfg is available, use it. Else use current cfg.teacher?
        # But current cfg.teacher might be for BIGRec if we are switching?
        # If distill_type is dllm2rec, we assume we want to distill FROM iLoRA.
        
        # Use teacher_cfg if available, else cfg
        t_cfg = teacher_cfg if teacher_cfg else cfg
        
        teacher_model_instance = create_teacher_model(
            t_cfg,
            num_items=dm.num_items,
            max_seq_len=t_cfg.student.max_seq_len,
            item_id_to_name=dm.mapped_id_to_title,
            padding_item_id=dm.padding_item_id,
            candidate_topk=t_cfg.distill.candidate_topk
        )
        
        loaded_teacher_trainer = iLoRATrainer.load_from_checkpoint(
            checkpoint_path=teacher_checkpoint_file_path,
            ilora_model=teacher_model_instance,
            num_items=dm.num_items,
            learning_rate=t_cfg.train.learning_rate,
            weight_decay=t_cfg.train.weight_decay,
            metrics_k=t_cfg.eval.metrics_k,
            item_id_to_name=dm.mapped_id_to_title,
            strict=False
        )
        teacher_model_instance = loaded_teacher_trainer.model
        for param in teacher_model_instance.parameters():
            param.requires_grad = False
        teacher_model_instance.eval()
        
        # Propensity Scores
        train_next_items = []
        for batch in dm.train_dataloader():
            train_next_items.extend(batch["next_item"].squeeze(-1).tolist())
        ps_calculator = PropensityScoreCalculator(
            item_num=dm.num_items + 1,
            train_next_items=train_next_items,
            power=cfg.distill.ps_power
        )
        propensity_scores = ps_calculator.get_ps()
        
        # Student
        student_model_instance = SASRec(
            num_items=dm.num_items,
            hidden_size=cfg.student.hidden_size,
            num_heads=cfg.student.num_heads,
            num_layers=cfg.student.num_layers,
            dropout_rate=cfg.student.dropout_rate,
            max_seq_len=cfg.student.max_seq_len,
            teacher_embedding_dim=teacher_model_instance.llm.config.hidden_size,
            padding_item_id=dm.padding_item_id
        )
        
        # DistillationTrainer
        distill_trainer = DistillationTrainer(
            student_model=student_model_instance,
            teacher_model=teacher_model_instance,
            datamodule=dm,
            num_items=dm.num_items,
            ranking_loss_weight=cfg.distill.ranking_loss_weight,
            embedding_loss_weight=cfg.distill.embedding_loss_weight,
            ce_loss_weight=cfg.distill.ce_loss_weight,
            ranking_temperature=cfg.distill.ranking_temperature,
            embedding_loss_type=cfg.distill.embedding_loss_type,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            metrics_k=cfg.eval.metrics_k,
            selection_policy=AllSamplesPolicy(),
            gamma_position=cfg.distill.gamma_position,
            gamma_confidence=cfg.distill.gamma_confidence,
            lambda_rank=cfg.distill.ranking_loss_weight,
            num_candidates=cfg.distill.candidate_topk,
            item_id_to_name=dm.mapped_id_to_title,
            alpha=cfg.distill.alpha,
            beta=cfg.distill.beta,
            propensity_scores=propensity_scores,
            lam=cfg.distill.lam,
            num_neg_samples=cfg.distill.num_neg_samples,
            teacher_output_dataloader=None # On-the-fly generation
        )

    # 9. PyTorch Lightning Trainerのインスタンス化と学習の実行
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs", version="")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="distilled-student-{epoch:02d}-{val_recall@10:.4f}",
        monitor=f"val_recall@{cfg.eval.metrics_k}",
        mode="max",
        save_top_k=1,
    )
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # progress_bar = CustomRichProgressBar()

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        precision="16-mixed" # Added for memory saving
    )

    logger.info("Starting distillation training...")
    trainer.fit(distill_trainer, datamodule=dm)
    logger.info("Distillation training finished.")

    # 10. 最終評価
    logger.info("Starting final evaluation of the distilled student model...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and Path(best_model_path).exists():
        logger.info(f"Loading best distilled model from: {best_model_path}")
        trainer.test(model=distill_trainer, datamodule=dm, ckpt_path=best_model_path)
    else:
        logger.warning("No best distilled model found. Testing with the last model state.")
        trainer.test(model=distill_trainer, datamodule=dm)

    logger.info(f"Distillation run finished. Results are in: {output_dir}")

def main():
    # --- Centralized Hydra Initialization ---
    overrides = sys.argv[1:]
    cfg = load_hydra_config(config_path="../../conf", overrides=overrides)
    run_experiment(cfg)

if __name__ == "__main__":
    main()