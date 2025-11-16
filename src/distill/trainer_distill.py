import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import logging
from tqdm import tqdm
import os
import random
from pathlib import Path

from src.student.models import get_student_model
from src.student.datamodule import StudentDataModule
from src.distill.data_bridge import DistillationDataBridge
from src.distill.kd_losses import compute_ranking_distillation_loss, compute_embedding_distillation_loss
from src.distill.selection_policy import get_selection_policy
from src.core.logging import TensorBoardLogger, time_block
from src.core.metrics import evaluate_metrics

log = logging.getLogger(__name__)

def train_distill_model(
    cfg: DictConfig,
    data_module: StudentDataModule,
    tb_logger: TensorBoardLogger,
    run_dir: Path,
    pre_trained_student_path: Path = None
):
    """
    Trains the student model with distillation.
    """
    log.info("Starting distillation training...")

    device = torch.device(cfg.general.device)
    
    # Initialize student model
    item_num = data_module.item_num
    max_seq_len = data_module.max_seq_len
    student_model = get_student_model(
        model_name=cfg.student.name,
        item_num=item_num,
        hidden_size=cfg.student.hidden_size,
        state_size=max_seq_len,
        dropout=cfg.student.dropout,
        device=device
    ).to(device)

    # Load pre-trained student model if path is provided
    if pre_trained_student_path and pre_trained_student_path.exists():
        log.info(f"Loading pre-trained student model from {pre_trained_student_path}")
        student_model.load_state_dict(torch.load(pre_trained_student_path, map_location=device))
    else:
        log.warning("No pre-trained student model found or provided. Training from scratch.")

    optimizer = optim.Adam(student_model.parameters(), lr=cfg.distill.lr)
    ce_criterion = nn.BCEWithLogitsLoss() # Cross-Entropy for main task

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    tb_writer = tb_logger.get_writer("distill")
    
    best_val_metric = -1.0
    patience_counter = 0
    
    model_save_path = run_dir / "models" / "distill"
    os.makedirs(model_save_path, exist_ok=True)

    # Load teacher outputs
    teacher_outputs_dir = Path(cfg.paths.teacher_outputs_dir)
    data_bridge = DistillationDataBridge(cfg, teacher_outputs_dir, device)
    data_bridge.load_teacher_outputs()

    # Initialize selection policy
    selection_policy = get_selection_policy(cfg)

    # Propensity scores for DROS (if alpha > 0)
    ps = None
    if cfg.distill.alpha > 0:
        # This needs to be calculated from the training data, similar to DLLM2Rec's main.py
        # For now, a dummy or a simplified calculation.
        # In DLLM2Rec, it's `calcu_propensity_score(train_data)`
        # We need `item_num` and `train_data` (dataframe) to calculate this.
        # For now, let's use a dummy `ps` if not properly calculated.
        log.warning("Propensity scores (ps) for DROS are not dynamically calculated yet. Using dummy.")
        ps = torch.rand(item_num, device=device)
        ps = ps / ps.sum() # Normalize

    with time_block("distill_student_train_time"):
        for epoch in range(cfg.distill.num_epochs):
            student_model.train()
            total_loss = 0
            total_ce_loss = 0
            total_kd_loss = 0
            total_rd_loss = 0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Distilling")):
                states = batch['seq'].to(device)
                len_states = batch['len_seq'].squeeze(1).to(device)
                targets = batch['target'].squeeze(1).to(device)
                original_indices = batch['original_index'].squeeze(1).to(device)

                optimizer.zero_grad()
                
                # Negative sampling (mimicking DLLM2Rec's main.py)
                num_negtive_items = 1 # As in DLLM2Rec baseline
                
                # Create a tensor of all items, then remove positive and history items
                # This is a bit complex to do efficiently for a whole batch.
                # For simplicity, let's assume we can sample negatives that are not the target.
                # A more robust way would be to use the `zeros_tensor` logic from DLLM2Rec.
                
                negative_items = []
                for t, s in zip(targets.cpu().numpy(), states.cpu().numpy()):
                    available_negatives = list(set(range(item_num)) - set([t]) - set(s))
                    if not available_negatives: # Fallback if no valid negatives
                        available_negatives = list(set(range(item_num)) - set([t]))
                    negative_items.append(torch.tensor(random.sample(available_negatives, num_negtive_items), device=device))
                negative_items = torch.stack(negative_items).squeeze(1) # (batch_size,)

                # Forward pass for student model
                # For ED, we need the student's item embeddings.
                # The `forward` method of student models returns logits.
                # We need to get the item embeddings from the student model to compute ED loss.
                # This might require modifying student models to expose embeddings, or
                # re-calculating them. For now, let's assume we can get them.
                
                # To get student item embeddings for ED, we need to pass `llm_all_embeddings`
                # to the student model's forward pass, which will then use it for ED.
                # The `ed_weight` parameter in student model's forward controls this.
                
                # Construct batch_llm_emb for ED
                batch_llm_emb = None
                if data_bridge.llm_all_embeddings is not None and cfg.distill.ed_weight > 0:
                    batch_llm_emb = torch.zeros(states.size(0), states.size(1), data_bridge.llm_all_embeddings.size(1), device=device)
                    valid_item_mask = (states < data_bridge.llm_all_embeddings.size(0))
                    batch_llm_emb[valid_item_mask] = data_bridge.llm_all_embeddings[states[valid_item_mask]]

                student_logits = student_model(states, len_states, llm_emb=batch_llm_emb, ed_weight=cfg.distill.ed_weight)

                # Compute CE Loss
                pos_scores = torch.gather(student_logits, 1, targets.unsqueeze(1))
                neg_scores = torch.gather(student_logits, 1, negative_items.unsqueeze(1))
                pos_labels = torch.ones_like(pos_scores)
                neg_labels = torch.zeros_like(neg_scores)
                scores = torch.cat((pos_scores, neg_scores), 0)
                labels = torch.cat((pos_labels, neg_labels), 0)
                ce_loss = ce_criterion(scores, labels)
                total_ce_loss += ce_loss.item()

                # Get teacher info for the current batch
                batch_teacher_info = data_bridge.get_batch_teacher_info(original_indices)
                
                # Student info for selection policy (if needed)
                student_info = {
                    "logits": student_logits,
                    # "embeddings": student_item_embeddings # If exposed by model
                }

                # Apply selection policy
                selection_mask = selection_policy.select(batch, batch_teacher_info, student_info)
                
                # Initialize KD loss components
                ranking_distillation_loss = torch.tensor(0.0, device=device)

                # Compute Ranking Distillation Loss if enabled and teacher rankings are available
                if cfg.distill.lambda_rd > 0 and batch_teacher_info.get('teacher_rankings') is not None:
                    # Filter batch_teacher_info and student_logits based on selection_mask
                    filtered_student_logits = student_logits[selection_mask]
                    filtered_teacher_rankings = batch_teacher_info['teacher_rankings'][selection_mask]
                    filtered_teacher_confidences = batch_teacher_info['teacher_confidences'][selection_mask]
                    filtered_negative_items = negative_items[selection_mask] # Use the same negative items

                    if filtered_student_logits.shape[0] > 0:
                        # Pass negative_items to the KD loss function
                        kd_cfg_with_neg = cfg.distill.copy()
                        kd_cfg_with_neg.negative_items = filtered_negative_items # Temporarily add to config for function
                        
                        ranking_distillation_loss = compute_ranking_distillation_loss(
                            student_logits=filtered_student_logits,
                            teacher_rankings=filtered_teacher_rankings,
                            teacher_confidences=filtered_teacher_confidences,
                            item_num=item_num,
                            cfg=kd_cfg_with_neg,
                            device=device,
                            ps=ps # Pass propensity scores
                        )
                        total_rd_loss += ranking_distillation_loss.item()

                # Compute Embedding Distillation Loss if enabled and teacher embeddings are available
                # Note: ED is applied in the student model's forward pass by adding `ed_weight * llm_emb`
                # So, the `compute_embedding_distillation_loss` here would be for an *additional* loss
                # if we wanted to enforce similarity between student's *learned* embeddings and teacher's.
                # The DLLM2Rec main.py applies ED directly in the forward pass.
                # So, we don't need an explicit `compute_embedding_distillation_loss` here for the main loss.
                # The `ed_weight` in the student model's forward pass already handles this.
                # If we wanted to add an *additional* loss, we would need student_item_embeddings.
                # For now, we follow DLLM2Rec's main.py and assume ED is handled in forward.
                
                # Total loss
                loss = ce_loss + cfg.distill.lambda_rd * ranking_distillation_loss
                total_loss += loss.item()
                total_kd_loss += (cfg.distill.lambda_rd * ranking_distillation_loss).item()

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_loader)
            avg_ce_loss = total_ce_loss / len(train_loader)
            avg_kd_loss = total_kd_loss / len(train_loader)
            avg_rd_loss = total_rd_loss / len(train_loader)

            log.info(f"Epoch {epoch+1}/{cfg.distill.num_epochs}, Total Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, KD Loss: {avg_kd_loss:.4f} (RD: {avg_rd_loss:.4f})")
            tb_writer.add_scalar("Loss/total_train", avg_loss, epoch)
            tb_writer.add_scalar("Loss/ce_train", avg_ce_loss, epoch)
            tb_writer.add_scalar("Loss/kd_train", avg_kd_loss, epoch)
            tb_writer.add_scalar("Loss/rd_train", avg_rd_loss, epoch)

            # Validation
            student_model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    states = batch['seq'].to(device)
                    len_states = batch['len_seq'].squeeze(1).to(device)
                    targets = batch['target'].squeeze(1).to(device)

                    # Construct batch_llm_emb for validation
                    batch_llm_emb = None
                    if data_bridge.llm_all_embeddings is not None and cfg.distill.ed_weight > 0:
                        batch_llm_emb = torch.zeros(states.size(0), states.size(1), data_bridge.llm_all_embeddings.size(1), device=device)
                        valid_item_mask = (states < data_bridge.llm_all_embeddings.size(0))
                        batch_llm_emb[valid_item_mask] = data_bridge.llm_all_embeddings[states[valid_item_mask]]

                    logits = student_model(states, len_states, llm_emb=batch_llm_emb, ed_weight=cfg.distill.ed_weight)
                    val_predictions.append(logits.cpu())
                    val_targets.append(targets.cpu())
            
            val_predictions = torch.cat(val_predictions, dim=0)
            val_targets = torch.cat(val_targets, dim=0)

            metrics = evaluate_metrics(val_predictions, val_targets, ks=[1, 5, 10, 20])
            log.info(f"Validation Metrics: {metrics}")
            for metric_name, score in metrics.items():
                tb_writer.add_scalar(f"Metrics/val/{metric_name}", score, epoch)
            
            current_val_metric = metrics.get(f"NDCG@{cfg.metrics.recsys.get('monitor_k', 20)}", metrics.get("NDCG@20", 0.0))

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
                # Save best model
                torch.save(student_model.state_dict(), os.path.join(model_save_path, "best_distill_model.pt"))
                log.info(f"New best validation metric ({best_val_metric:.4f}). Model saved.")
            else:
                patience_counter += 1
                log.info(f"Validation metric did not improve. Patience: {patience_counter}/{cfg.distill.get('patience', 10)}")
                if patience_counter >= cfg.distill.get('patience', 10):
                    log.info("Early stopping triggered.")
                    break
    
    log.info("Distillation training finished.")
    tb_writer.close()

if __name__ == "__main__":
    # Example usage with dummy config and data
    from omegaconf import OmegaConf
    from src.core.paths import get_data_dir, get_current_run_dir
    from src.core.data_utils import preprocess_data
    from src.core.logging import setup_logging
    from src.core.seed import set_seed
    import shutil
    from pathlib import Path
    import numpy as np # Added import for numpy

    # Setup logging
    log_file_path = Path("temp_run_distill/logs/train_distill.log")
    setup_logging(log_file_path)

    # Dummy config
    dummy_cfg = OmegaConf.create({
        "general": {
            "seed": 42,
            "device": "cpu" # Use CPU for example
        },
        "dataset": {
            "name": "test_dataset",
            "max_seq_len": 50,
            "min_user_inter": 5,
            "split": {"method": "leave_one_out"}
        },
        "paths": {
            "data_dir": "temp_data_distill",
            "result_root": "temp_run_distill",
            "teacher_outputs_dir": "temp_teacher_outputs_distill" # Dummy teacher outputs
        },
        "student": {
            "name": "SASRec",
            "emb_size": 64,
            "hidden_size": 64,
            "max_seq_len": 50,
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 1e-3,
            "num_epochs": 3, # Reduced for quick test
            "patience": 2
        },
        "distill": {
            "name": "dllm2rec",
            "alpha": 0.5,       # ranking distill の重み
            "ed_weight": 0.3,   # embedding distill の重み
            "lambda": 0.7,      # Weight for ranking distillation loss (lam in DLLM2Rec)
            "batch_size": 32,
            "lr": 1e-3,
            "num_epochs": 3,    # Reduced for quick test
            "gamma_position": 0.3,
            "gamma_confidence": 0.5,
            "gamma_consistency": 0.1,
            "patience": 2
        },
        "active": {
            "name": "all",
            "strategy": "all",
            "budget_ratio": 1.0
        },
        "metrics": {
            "recsys": {
                "monitor_k": 20
            }
        },
        "hydra": {
            "run": {
                "dir": "temp_run_distill"
            }
        }
    })
    
    # Set seed
    set_seed(dummy_cfg.general.seed)

    # Prepare dummy data
    temp_data_dir = get_data_dir(dummy_cfg)
    preprocess_data(dummy_cfg, temp_data_dir)

    # Create dummy teacher output files
    temp_teacher_outputs_dir = Path(dummy_cfg.paths.teacher_outputs_dir)
    temp_teacher_outputs_dir.mkdir(parents=True, exist_ok=True)

    item_num_dummy = 1000
    llm_emb_dim_dummy = 4096
    num_train_samples_dummy = 100 # Should match actual train_df size
    top_n_dummy = 10 # From distill.candidate_topk

    dummy_llm_all_embeddings = torch.randn(item_num_dummy + 1, llm_emb_dim_dummy) # +1 for padding
    dummy_teacher_rankings = torch.randint(0, item_num_dummy, (num_train_samples_dummy, top_n_dummy))
    dummy_teacher_confidences = torch.rand(num_train_samples_dummy, top_n_dummy)

    torch.save(dummy_llm_all_embeddings, temp_teacher_outputs_dir / "all_embeddings.pt")
    np.savetxt(temp_teacher_outputs_dir / "myrank_train.txt", dummy_teacher_rankings.numpy(), fmt='%d')
    np.savetxt(temp_teacher_outputs_dir / "confidence_train.txt", dummy_teacher_confidences.numpy(), fmt='%.6f')

    # Setup data module
    device = torch.device(dummy_cfg.general.device)
    data_module = StudentDataModule(dummy_cfg, temp_data_dir, device)
    data_module.prepare_data()
    data_module.setup()

    # Setup TensorBoard Logger
    run_dir = Path(get_current_run_dir(dummy_cfg))
    tb_log_dir = run_dir / "tb"
    tb_logger = TensorBoardLogger(tb_log_dir)

    # Set time_block output path
    time_block.set_output_path(run_dir / "metrics" / "time.json")

    # Run distillation training
    train_distill_model(dummy_cfg, data_module, tb_logger, run_dir)

    # Clean up
    shutil.rmtree("temp_data_distill", ignore_errors=True)
    shutil.rmtree("temp_teacher_outputs_distill", ignore_errors=True)
    shutil.rmtree("temp_run_distill", ignore_errors=True)
    print("\nCleaned up temporary directories.")
