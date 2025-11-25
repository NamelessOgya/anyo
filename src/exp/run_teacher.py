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
    logger.error("THIS IS THE LATEST CODE (Version 20251124)")
    # --- Manual Hydra Initialization ---
    # Parse command-line overrides manually
    overrides = sys.argv[1:]
    
    # Manual override parsing for llm_model config
    llm_model_override = None
    new_overrides = []
    for o in overrides:
        # Handle both `llm_model=...` and `+llm_model=...`
        if "llm_model=" in o:
            llm_model_override = o.split("=")[1]
        else:
            new_overrides.append(o)
    overrides = new_overrides

    with hydra.initialize(config_path="../../conf", version_base="1.3", job_name="teacher_run"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    # Apply llm_model override if found
    if llm_model_override:
        llm_model_cfg_path = get_project_root() / f"conf/llm_model/{llm_model_override}.yaml"
        if not llm_model_cfg_path.exists():
            raise FileNotFoundError(f"LLM model config not found: {llm_model_cfg_path}")
        llm_model_cfg = OmegaConf.load(llm_model_cfg_path)
        
        # Manually update config values
        cfg.teacher.llm_model_name = llm_model_cfg.llm_model_name
        cfg.train.batch_size = llm_model_cfg.batch_size
        logger.info(f"LLM model override applied: {llm_model_override}")
        logger.info(f"Updated cfg.teacher.llm_model_name: {cfg.teacher.llm_model_name}")
        logger.info(f"Updated cfg.train.batch_size: {cfg.train.batch_size}")

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

    # Initialize tokenizer early
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name, use_fast=False)
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm_tokenizer.padding_side = "right"

    # DEBUG: Tokenizer Test Start
    logger.info("DEBUG(run_teacher): Starting tokenizer test for [HistoryEmb].")
    his_token_id_expected = llm_tokenizer.additional_special_tokens_ids[llm_tokenizer.additional_special_tokens.index("[HistoryEmb]")]
    logger.info(f"DEBUG(run_teacher): Expected [HistoryEmb] ID: {his_token_id_expected}")

    test_string_hist_count = 5
    test_string = "".join(["[HistoryEmb]"] * test_string_hist_count)
    tokenized_test = llm_tokenizer(test_string, return_tensors="pt", add_special_tokens=False) # Llama tokenizer requires add_special_tokens=False for raw string without prefix
    
    logger.info(f"DEBUG(run_teacher): Test string: '{test_string}'")
    logger.info(f"DEBUG(run_teacher): Tokenized test IDs: {tokenized_test['input_ids']}")
    logger.info(f"DEBUG(run_teacher): Tokenized test tokens: {llm_tokenizer.convert_ids_to_tokens(tokenized_test['input_ids'][0].tolist())}")
    
    num_his_tokens_in_test = (tokenized_test['input_ids'][0] == his_token_id_expected).sum().item()
    logger.info(f"DEBUG(run_teacher): Number of [HistoryEmb] tokens found in test string tokenization: {num_his_tokens_in_test}")
    if num_his_tokens_in_test == test_string_hist_count:
        logger.info("DEBUG(run_teacher): Tokenizer correctly identifies [HistoryEmb] as single tokens.")
    else:
        logger.error(f"DEBUG(run_teacher): Tokenizer MISIDENTIFIES [HistoryEmb]. Expected {test_string_hist_count}, got {num_his_tokens_in_test}.")
    logger.info("DEBUG(run_teacher): Tokenizer Test End.")
    # DEBUG: Tokenizer Test End


    # 2. SASRecDataModuleのインスタンス化とデータ準備
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        tokenizer=llm_tokenizer, # Pass tokenizer
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv",
        seed=cfg.seed
    )
    dm.prepare_data()
    dm.setup()

    # 3. create_teacher_model を使用して iLoRAModel をインスタンス化
    ilora_model_instance = create_teacher_model(
        cfg,
        llm_tokenizer=llm_tokenizer,
        num_items=dm.num_items,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=dm.mapped_id_to_title,
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
        item_id_to_name=dm.mapped_id_to_title
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
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches
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
            llm_tokenizer=llm_tokenizer,
            num_items=dm.num_items,
            max_seq_len=cfg.student.max_seq_len,
            item_id_to_name=dm.mapped_id_to_title,
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
            item_id_to_name=dm.mapped_id_to_title
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
    if not cfg.eval.get("skip_test", False):
        logger.info("Evaluating on test set with the best model...")
        test_start_time = time.perf_counter() # Measure test duration
        if best_model_path and Path(best_model_path).exists():
            trainer.test(model=trainer_model_for_eval, datamodule=dm) # Use the potentially new instance
        else:
            logger.warning("No best model found. Testing with the last model state.")
            trainer.test(model=trainer_model_for_eval, datamodule=dm) # Use the potentially new instance
        test_duration = time.perf_counter() - test_start_time
        logger.info(f"Test set evaluation finished. Duration: {test_duration:.2f} seconds.")
    else:
        logger.info("Skipping test set evaluation as per configuration.")

    # 6.2. 訓練データセットに対する教師出力を生成し、保存
    if not cfg.teacher.get("skip_output_generation", False):
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
        # Ensure to pass collate_fn if the datamodule is using it
        inference_dataloader = DataLoader(
            dm.train_dataset,
            batch_size=cfg.teacher.inference_batch_size, # Use inference_batch_size
            shuffle=False, # No need to shuffle for inference
            num_workers=cfg.train.num_workers, # Use the same num_workers as training
            collate_fn=dm.collater if hasattr(dm, 'collater') else None # Pass collate_fn
        )

        batch_idx = 0
        all_ranking_scores = []
        all_embeddings = []
        all_candidates = []
        all_confidence = []
        
        save_interval = cfg.teacher.get("save_output_interval", 100) # Default to 100 batches

        with torch.no_grad():
            for batch in inference_dataloader:
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                teacher_outputs = trainer_model_for_eval.model.get_teacher_outputs(batch)
                
                all_ranking_scores.append(teacher_outputs["ranking_scores"].cpu())
                all_embeddings.append(teacher_outputs["embeddings"].cpu())
                all_candidates.append(teacher_outputs["candidates"].cpu())
                all_confidence.append(teacher_outputs["confidence"].cpu())
                
                batch_idx += 1
                if batch_idx % save_interval == 0:
                    current_chunk_idx = batch_idx // save_interval
                    chunk_output_path = teacher_outputs_batches_dir / f"chunk_{current_chunk_idx:05d}.pt"
                    
                    torch.save({
                        "ranking_scores": torch.cat(all_ranking_scores, dim=0),
                        "embeddings": torch.cat(all_embeddings, dim=0),
                        "candidates": torch.cat(all_candidates, dim=0),
                        "confidence": torch.cat(all_confidence, dim=0),
                    }, chunk_output_path)
                    
                    logger.info(f"Generated and saved outputs for {batch_idx} batches (chunk {current_chunk_idx}).")
                    
                    all_ranking_scores = []
                    all_embeddings = []
                    all_candidates = []
                    all_confidence = []
        
            # Save any remaining outputs
            if len(all_ranking_scores) > 0:
                current_chunk_idx = batch_idx // save_interval + (1 if batch_idx % save_interval != 0 else 0)
                chunk_output_path = teacher_outputs_batches_dir / f"chunk_{current_chunk_idx:05d}_final.pt"
                torch.save({
                    "ranking_scores": torch.cat(all_ranking_scores, dim=0),
                    "embeddings": torch.cat(all_embeddings, dim=0),
                    "candidates": torch.cat(all_candidates, dim=0),
                    "confidence": torch.cat(all_confidence, dim=0),
                }, chunk_output_path)
                logger.info(f"Generated and saved remaining outputs for {len(all_ranking_scores)} batches (final chunk {current_chunk_idx}).")
                
        teacher_output_generation_duration = time.perf_counter() - teacher_output_generation_start_time
        logger.info(f"Teacher outputs for {batch_idx} total batches saved to {teacher_outputs_batches_dir}. Duration: {teacher_output_generation_duration:.2f} seconds.")
    else:
        logger.info("Skipping teacher output generation as per configuration.")

    logger.info(f"Teacher model run finished. Results are in: {output_dir}")

if __name__ == "__main__":
    main()