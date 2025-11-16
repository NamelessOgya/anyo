import torch
from omegaconf import DictConfig
import logging
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict

from src.student.models import get_student_model
from src.student.datamodule import StudentDataModule
from src.core.metrics import evaluate_metrics
from src.core.logging import time_block

log = logging.getLogger(__name__)

def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    device: torch.device,
    output_path: Path,
    llm_all_emb: torch.Tensor = None,
    ed_weight: float = 0.0
) -> Dict[str, float]:
    """
    Evaluates a given model on a data loader and saves metrics to a JSON file.
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            states = batch['seq'].to(device)
            len_states = batch['len_seq'].squeeze(1).to(device)
            batch_targets = batch['target'].squeeze(1).to(device)

            # If llm_all_emb is provided, we need to construct llm_emb for the batch
            batch_llm_emb = None
            if llm_all_emb is not None:
                # Create a tensor of zeros with the shape of (batch_size, max_seq_len, llm_emb_dim)
                batch_llm_emb = torch.zeros(states.size(0), states.size(1), llm_all_emb.size(1), device=device)
                # Mask for valid item IDs (not padding)
                valid_item_mask = (states < llm_all_emb.size(0))
                # Fill batch_llm_emb with corresponding llm_all_emb for valid items
                batch_llm_emb[valid_item_mask] = llm_all_emb[states[valid_item_mask]]

            logits = model(states, len_states, llm_emb=batch_llm_emb, ed_weight=ed_weight)
            predictions.append(logits.cpu())
            targets.append(batch_targets.cpu())
    
    all_predictions = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)

    ks = cfg.metrics.recsys.get('ks', [1, 5, 10, 20]) # Default K values
    metrics = evaluate_metrics(all_predictions, all_targets, ks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    log.info(f"Evaluation metrics saved to {output_path}")
    for metric_name, score in metrics.items():
        log.info(f"  {metric_name}: {score:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Example usage with dummy config and data
    from omegaconf import OmegaConf
    from src.core.paths import get_data_dir, get_current_run_dir
    from src.core.data_utils import preprocess_data
    from src.core.logging import setup_logging
    from src.core.seed import set_seed
    import shutil
    from pathlib import Path

    # Setup logging
    log_file_path = Path("temp_run/logs/evaluator.log")
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
            "data_dir": "temp_data",
            "result_root": "temp_run"
        },
        "student": {
            "name": "SASRec",
            "emb_size": 64,
            "hidden_size": 64,
            "max_seq_len": 50,
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 1e-3,
            "num_epochs": 3,
            "patience": 2
        },
        "metrics": {
            "recsys": {
                "ks": [1, 5, 10],
                "monitor_k": 10
            }
        },
        "hydra": {
            "run": {
                "dir": "temp_run"
            }
        }
    })
    
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

    # Create a dummy model and load some weights (for demonstration)
    item_num = data_module.item_num
    max_seq_len = data_module.max_seq_len
    dummy_model = get_student_model(
        model_name=dummy_cfg.student.name,
        item_num=item_num,
        hidden_size=dummy_cfg.student.hidden_size,
        state_size=max_seq_len,
        dropout=dummy_cfg.student.dropout,
        device=device
    ).to(device)
    # Simulate loading a trained model
    # For a real scenario, you'd load from a saved checkpoint
    # For this test, we'll just use random weights.

    # Dummy LLM embeddings (simulate teacher output)
    dummy_llm_all_emb = torch.randn(item_num + 1, 4096, device=device) # +1 for padding item

    # Evaluate the dummy model
    run_dir = get_current_run_dir(dummy_cfg)
    metrics_output_path = Path(run_dir) / "metrics" / "eval_baseline.json"
    log.info("Evaluating dummy model...")
    with time_block("eval_time"):
        evaluate_model(dummy_model, data_module.test_dataloader(), dummy_cfg, device, metrics_output_path,
                       llm_all_emb=dummy_llm_all_emb, ed_weight=0.5)

    # Clean up
    shutil.rmtree("temp_data", ignore_errors=True)
    shutil.rmtree("temp_run", ignore_errors=True)
    print("\nCleaned up temporary directories.")
