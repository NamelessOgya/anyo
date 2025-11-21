import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import time # Added import for time
from pathlib import Path
from datetime import datetime
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info
from src.core.callbacks import CustomRichProgressBar

from src.student.datamodule import SASRecDataModule # 教師モデルも同じデータモジュールを使用
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer
from src.student.evaluator import SASRecEvaluator # 評価は生徒モデルの評価器を流用
from src.teacher.mlp_projector import MLPProjector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
from torch.utils.data import DataLoader # Added DataLoader import

logger = logging.getLogger(__name__)

def main():
    # --- Manual Hydra Initialization ---
    # Parse command-line overrides manually
    overrides = sys.argv[1:]
    
    with hydra.initialize(config_path="../../conf", version_base="1.3", job_name="teacher_run"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    # --- End Manual Hydra Initialization ---

    # 1. ロギング、シード、Git情報の初期化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_project_root() / "result" / f"teacher_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"!!! SCRIPT RUNNING. MANUALLY CREATED OUTPUT DIR: {output_dir} !!!")
    
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

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
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 3. create_teacher_model を使用して iLoRAModel をインスタンス化
    ilora_model_instance = create_teacher_model(
        cfg,
        num_items=dm.num_items,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=dm.item_id_to_name,
        padding_item_id=dm.padding_item_id,
        candidate_topk=cfg.distill.candidate_topk
    )

    # 4. iLoRATrainerのインスタンス化
    trainer_model = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k,
        item_id_to_name=dm.item_id_to_name
    )

    # 5. PyTorch Lightning Trainerのセットアップ
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs", version="")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="teacher-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = CustomRichProgressBar()

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps,
        precision=cfg.train.precision
    )

    logger.info("Starting iLoRA teacher model training...")
    training_start_time = time.perf_counter() # Measure training duration
    trainer.fit(trainer_model, datamodule=dm)
    training_duration = time.perf_counter() - training_start_time
    logger.info(f"iLoRA teacher model training finished. Duration for trainer.fit: {training_duration:.2f} seconds.")

    # 6. 評価と教師出力の生成
    logger.info("Starting final evaluation and teacher output generation...")

    # Load the best model for testing and teacher output generation
    trainer_model_for_eval = None
    best_model_path = checkpoint_callback.best_model_path

    if cfg.teacher.get("use_qlora", False) and best_model_path and Path(best_model_path).exists():
        logger.info(f"Loading best QLoRA model from {best_model_path} for evaluation and output generation.")
        # Create a new iLoRAModel instance with QLoRA enabled
        eval_ilora_model_instance = create_teacher_model(
            cfg,
            num_items=dm.num_items,
            max_seq_len=cfg.student.max_seq_len,
            item_id_to_name=dm.item_id_to_name,
            padding_item_id=dm.padding_item_id,
            candidate_topk=cfg.distill.candidate_topk
        )
        # Create a new iLoRATrainer instance to load the checkpoint into
        eval_trainer_model = iLoRATrainer(
            ilora_model=eval_ilora_model_instance,
            num_items=dm.num_items,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            metrics_k=cfg.eval.metrics_k,
            item_id_to_name=dm.item_id_to_name
        )
        # Load the checkpoint
        device = "cuda" if torch.cuda.is_available() and cfg.train.accelerator == "gpu" else "cpu"
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Load the state_dict for the entire trainer_model with strict=False
        # This will load all non-LLM parameters and any LLM parameters that match
        eval_trainer_model.load_state_dict(checkpoint['state_dict'], strict=False)

        # The first load_state_dict with strict=False should handle restoring the
        # trainable LoRA weights. The second, stricter load for the LLM is causing
        # issues with QLoRA's state dict keys and is likely unnecessary.
        #
        # # Now, explicitly load the state_dict for the PEFT-wrapped LLM
        # # We need to filter the checkpoint's state_dict to get only LLM-related keys
        # llm_state_dict_from_checkpoint = {
        #     key.replace('model.llm.', ''): value
        #     for key, value in checkpoint['state_dict'].items()
        #     if key.startswith('model.llm.')
        # }
        # # Assuming eval_trainer_model.model.llm is already a PeftModel
        # eval_trainer_model.model.llm.load_state_dict(llm_state_dict_from_checkpoint, strict=True)
        
        logger.info(f"Loaded best model from {best_model_path} for evaluation and output generation.")
        trainer_model_for_eval = eval_trainer_model
    else:
        logger.info("Using training model instance for evaluation and output generation (QLoRA disabled or no best checkpoint).")
        trainer_model_for_eval = trainer_model # Use the training instance if no QLoRA or no best checkpoint

    # 6.1. テストセットでの評価
    logger.info("Evaluating on test set with the best model...")
    test_start_time = time.perf_counter() # Measure test duration
    if best_model_path and Path(best_model_path).exists():
        trainer.test(model=trainer_model_for_eval, datamodule=dm) # Use the potentially new instance
    else:
        logger.warning("No best model found. Testing with the last model state.")
        trainer.test(model=trainer_model_for_eval, datamodule=dm) # Use the potentially new instance
    test_duration = time.perf_counter() - test_start_time
    logger.info(f"Test set evaluation finished. Duration: {test_duration:.2f} seconds.")

    # 6.2. 訓練データセットに対する教師出力を生成し、保存
    # 手動ループのために、モデル全体を正しいデバイスに移動させ、評価モードに設定します。
    logger.info("Preparing for teacher output generation...")
    teacher_output_generation_start_time = time.perf_counter() # Measure teacher output generation duration
    device = "cuda" if torch.cuda.is_available() and cfg.train.accelerator == "gpu" else "cpu"
    trainer_model_for_eval.to(device) # Use the potentially new instance
    trainer_model_for_eval.eval()

    logger.info("Generating teacher outputs for the training dataset...")
    teacher_outputs_batches_dir = output_dir / "teacher_outputs_batches"
    teacher_outputs_batches_dir.mkdir(parents=True, exist_ok=True)

    # Use a separate DataLoader with inference_batch_size for teacher output generation
    inference_dataloader = DataLoader(
        dm.train_dataset,
        batch_size=cfg.teacher.inference_batch_size, # Use inference_batch_size
        shuffle=False, # No need to shuffle for inference
        num_workers=cfg.train.num_workers # Use the same num_workers as training
    )

    batch_idx = 0
    with torch.no_grad():
        for batch in inference_dataloader:
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            teacher_outputs = trainer_model_for_eval.model.get_teacher_outputs(batch) # Use the potentially new instance
            
            # Save each batch's output to a separate file
            batch_output_path = teacher_outputs_batches_dir / f"batch_{batch_idx:05d}.pt"
            torch.save({
                "ranking_scores": teacher_outputs["ranking_scores"].cpu(),
                "embeddings": teacher_outputs["embeddings"].cpu(),
                "candidates": teacher_outputs["candidates"].cpu(),
                "confidence": teacher_outputs["confidence"].cpu(),
            }, batch_output_path)
            batch_idx += 1
            if batch_idx % 10 == 0:
                logger.info(f"Generated outputs for {batch_idx} batches...")
            
    teacher_output_generation_duration = time.perf_counter() - teacher_output_generation_start_time
    logger.info(f"Teacher outputs for {batch_idx} batches saved to {teacher_outputs_batches_dir}. Duration: {teacher_output_generation_duration:.2f} seconds.")

    logger.info(f"Teacher model run finished. Results are in: {output_dir}")

if __name__ == "__main__":
    main()