import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import logging
from tqdm import tqdm
import os

from src.student.models import get_student_model
from src.student.datamodule import StudentDataModule
from src.core.logging import TensorBoardLogger, time_block
from src.core.metrics import evaluate_metrics

log = logging.getLogger(__name__)


def train_student_baseline(
    cfg: DictConfig,
    data_module: StudentDataModule,
    tb_logger: TensorBoardLogger,
    run_dir: str,
):
    """
    Trains the student model without distillation (baseline).
    """
    log.info("Starting student baseline training...")

    device = torch.device(cfg.general.device)

    # Initialize model
    item_num = data_module.item_num
    max_seq_len = data_module.max_seq_len
    student_model = get_student_model(
        model_name=cfg.student.name,
        item_num=item_num,
        hidden_size=cfg.student.hidden_size,
        state_size=max_seq_len,
        dropout=cfg.student.dropout,
        device=device,
    ).to(device)

    optimizer = optim.Adam(student_model.parameters(), lr=cfg.student.lr)
    criterion = nn.BCEWithLogitsLoss()  # As used in DLLM2Rec

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    tb_writer = tb_logger.get_writer("baseline")

    best_val_metric = -1.0
    patience_counter = 0

    model_save_path = os.path.join(run_dir, "models", "baseline")
    os.makedirs(model_save_path, exist_ok=True)

    with time_block("student_baseline_train_time"):
        for epoch in range(cfg.student.num_epochs):
            student_model.train()
            total_loss = 0
            for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
            ):
                states = batch["seq"].to(device)
                len_states = batch["len_seq"].squeeze(1).to(device)
                targets = batch["target"].squeeze(1).to(device)  # (batch_size,)

                optimizer.zero_grad()

                # Forward pass
                # For baseline, llm_emb is None and ed_weight is 0
                logits = student_model(states, len_states, llm_emb=None, ed_weight=0.0)

                # Prepare targets for BCEWithLogitsLoss
                # We need to create one-hot like labels for positive and negative samples
                # DLLM2Rec uses a specific negative sampling strategy and BCE loss
                # For simplicity in baseline, we'll assume a single positive target per sequence
                # and treat all other items as negative for a multi-label BCE loss,
                # or adapt to the DLLM2Rec's positive/negative sampling if needed.
                # For now, let's mimic DLLM2Rec's BCE loss with explicit pos/neg.

                # Negative sampling (simplified for baseline)
                # In DLLM2Rec, negative items are sampled per batch.
                # Here, we'll just use a simple approach for baseline:
                # For each positive target, sample one negative item.
                # This is a simplification and might need adjustment to match DLLM2Rec's exact setup.

                # For now, let's use the DLLM2Rec's approach of sampling one negative item per positive
                # This requires knowing the item_num to sample from.

                # Simplified BCE loss: treat target as positive, others as negative
                # This is a common way to use BCE for ranking, but DLLM2Rec does explicit neg sampling.
                # Let's try to replicate DLLM2Rec's BCE loss structure for consistency.

                # Create a mask for positive and negative items
                pos_labels = torch.zeros_like(logits, dtype=torch.float)
                pos_labels.scatter_(
                    1, targets.unsqueeze(1), 1.0
                )  # Set target item to 1

                # For negative sampling, we need to exclude the positive item
                # and potentially other items in the sequence history.
                # For baseline, we can simplify: assume all other items are negative.
                # This is effectively a multi-label classification where only one label is positive.
                # However, DLLM2Rec's main.py uses explicit pos/neg scores.

                # Let's use the DLLM2Rec's explicit positive and negative sampling for consistency.
                # This means we need to sample negative items here.

                # Negative sampling (mimicking DLLM2Rec's main.py)
                num_negtive_items = 1  # As in DLLM2Rec baseline

                # Create a tensor of all items, then remove positive and history items
                # This is a bit complex to do efficiently for a whole batch.
                # For simplicity, let's assume we can sample negatives that are not the target.
                # A more robust way would be to use the `zeros_tensor` logic from DLLM2Rec.

                # For now, a simple negative sampling:
                negative_items = []
                for t, s in zip(targets.cpu().numpy(), states.cpu().numpy()):
                    available_negatives = list(set(range(item_num)) - set([t]) - set(s))
                    if not available_negatives:  # Fallback if no valid negatives
                        available_negatives = list(set(range(item_num)) - set([t]))
                    negative_items.append(
                        torch.tensor(
                            random.sample(available_negatives, num_negtive_items),
                            device=device,
                        )
                    )
                negative_items = torch.stack(negative_items).squeeze(1)  # (batch_size,)

                pos_scores = torch.gather(
                    logits, 1, targets.unsqueeze(1)
                )  # (batch_size, 1)
                neg_scores = torch.gather(
                    logits, 1, negative_items.unsqueeze(1)
                )  # (batch_size, 1)

                pos_labels = torch.ones_like(pos_scores)
                neg_labels = torch.zeros_like(neg_scores)

                scores = torch.cat((pos_scores, neg_scores), 0)
                labels = torch.cat((pos_labels, neg_labels), 0)

                loss = criterion(scores, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            log.info(
                f"Epoch {epoch+1}/{cfg.student.num_epochs}, Train Loss: {avg_loss:.4f}"
            )
            tb_writer.add_scalar("Loss/train", avg_loss, epoch)

            # Validation
            student_model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    states = batch["seq"].to(device)
                    len_states = batch["len_seq"].squeeze(1).to(device)
                    targets = batch["target"].squeeze(1).to(device)

                    logits = student_model(
                        states, len_states, llm_emb=None, ed_weight=0.0
                    )
                    val_predictions.append(logits.cpu())
                    val_targets.append(targets.cpu())

            val_predictions = torch.cat(val_predictions, dim=0)
            val_targets = torch.cat(val_targets, dim=0)

            metrics = evaluate_metrics(val_predictions, val_targets, ks=[1, 5, 10, 20])
            log.info(f"Validation Metrics: {metrics}")
            for metric_name, score in metrics.items():
                tb_writer.add_scalar(f"Metrics/val/{metric_name}", score, epoch)

            current_val_metric = metrics.get(
                f"NDCG@{cfg.metrics.recsys.get('monitor_k', 20)}",
                metrics.get("NDCG@20", 0.0),
            )  # Default to NDCG@20

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
                # Save best model
                torch.save(
                    student_model.state_dict(),
                    os.path.join(model_save_path, "best_baseline_model.pt"),
                )
                log.info(
                    f"New best validation metric ({best_val_metric:.4f}). Model saved."
                )
            else:
                patience_counter += 1
                log.info(
                    f"Validation metric did not improve. Patience: {patience_counter}/{cfg.student.get('patience', 10)}"
                )
                if patience_counter >= cfg.student.get("patience", 10):
                    log.info("Early stopping triggered.")
                    break

    log.info("Student baseline training finished.")
    tb_writer.close()


if __name__ == "__main__":
    # Example usage with dummy config and data
    from omegaconf import OmegaConf
    from src.core.paths import get_data_dir, get_current_run_dir
    from src.core.data_utils import preprocess_data
    from src.core.logging import setup_logging, time_block
    from src.core.seed import set_seed
    import random
    import shutil
    from pathlib import Path

    # Setup logging
    log_file_path = Path("temp_run/logs/train_baseline.log")
    setup_logging(log_file_path)

    # Dummy config
    dummy_cfg = OmegaConf.create(
        {
            "general": {"seed": 42, "device": "cpu"},  # Use CPU for example
            "dataset": {
                "name": "test_dataset",
                "max_seq_len": 50,
                "min_user_inter": 5,
                "split": {"method": "leave_one_out"},
            },
            "paths": {"data_dir": "temp_data", "result_root": "temp_run"},
            "student": {
                "name": "SASRec",
                "emb_size": 64,
                "hidden_size": 64,
                "max_seq_len": 50,
                "dropout": 0.1,
                "batch_size": 32,
                "lr": 1e-3,
                "num_epochs": 3,  # Reduced for quick test
                "patience": 2,
            },
            "metrics": {"recsys": {"monitor_k": 20}},
            "hydra": {"run": {"dir": "temp_run"}},
        }
    )

    # Set seed
    set_seed(dummy_cfg.general.seed)

    # Prepare dummy data
    temp_data_dir = get_data_dir(dummy_cfg)
    preprocess_data(dummy_cfg, temp_data_dir)

    # Setup data module
    device = torch.device(dummy_cfg.general.device)
    data_module = StudentDataModule(dummy_cfg, temp_data_dir, device)
    data_module.prepare_data()
    data_module.setup()

    # Setup TensorBoard Logger
    tb_log_dir = Path(get_current_run_dir(dummy_cfg)) / "tb"
    tb_logger = TensorBoardLogger(tb_log_dir)

    # Set time_block output path
    time_block.set_output_path(
        Path(get_current_run_dir(dummy_cfg)) / "metrics" / "time.json"
    )

    # Run training
    train_student_baseline(
        dummy_cfg, data_module, tb_logger, get_current_run_dir(dummy_cfg)
    )

    # Clean up
    shutil.rmtree("temp_data", ignore_errors=True)
    shutil.rmtree("temp_run", ignore_errors=True)
    print("\nCleaned up temporary directories.")
