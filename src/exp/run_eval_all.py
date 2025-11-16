import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import torch
import json

from src.core.logging import setup_logging, time_block
from src.core.seed import set_seed
from src.core.git_info import save_git_info
from src.core.paths import get_current_run_dir, get_data_dir, get_teacher_outputs_dir
from src.core.data_utils import preprocess_data
from src.student.datamodule import StudentDataModule
from src.student.evaluator import evaluate_model
from src.student.models import get_student_model
from src.distill.data_bridge import DistillationDataBridge

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Resolve paths early
    run_dir = Path(get_current_run_dir(cfg))
    data_dir = Path(get_data_dir(cfg))
    teacher_outputs_dir = Path(get_teacher_outputs_dir(cfg))

    # Setup logging
    log_file = run_dir / "logs" / "run_eval_all.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file)
    log.info(f"Starting all evaluation run in {run_dir}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    set_seed(cfg.general.seed)

    # Save git info
    save_git_info(run_dir / "metrics" / "git_info.txt")

    # Set time_block output path
    time_block.set_output_path(run_dir / "metrics" / "time.json")

    # Prepare data (e.g., download, preprocess)
    preprocess_data(cfg, str(data_dir))

    try:
        device = torch.device(cfg.general.device)
        data_module = StudentDataModule(cfg, str(data_dir), device)
        data_module.prepare_data()
        data_module.setup(stage="test")  # Only need test set for evaluation

        all_results = {}

        # --- Evaluate Baseline Model ---
        log.info("Evaluating baseline model...")
        baseline_model_path = run_dir / "models" / "baseline" / "best_baseline_model.pt"
        if baseline_model_path.exists():
            student_model_baseline = get_student_model(
                model_name=cfg.student.name,
                item_num=data_module.item_num,
                hidden_size=cfg.student.hidden_size,
                state_size=data_module.max_seq_len,
                dropout=cfg.student.dropout,
                device=device,
            ).to(device)
            student_model_baseline.load_state_dict(
                torch.load(baseline_model_path, map_location=device)
            )

            metrics_output_path_baseline = run_dir / "metrics" / "eval_baseline.json"
            with time_block("eval_baseline_time"):
                baseline_metrics = evaluate_model(
                    student_model_baseline,
                    data_module.test_dataloader(),
                    cfg,
                    device,
                    metrics_output_path_baseline,
                )
            all_results["baseline"] = baseline_metrics
        else:
            log.warning(
                f"Baseline model not found at {baseline_model_path}. Skipping baseline evaluation."
            )

        # --- Evaluate Distilled Model ---
        log.info("Evaluating distilled model...")
        distill_model_path = run_dir / "models" / "distill" / "best_distill_model.pt"
        if distill_model_path.exists():
            student_model_distill = get_student_model(
                model_name=cfg.student.name,
                item_num=data_module.item_num,
                hidden_size=cfg.student.hidden_size,
                state_size=data_module.max_seq_len,
                dropout=cfg.student.dropout,
                device=device,
            ).to(device)
            student_model_distill.load_state_dict(
                torch.load(distill_model_path, map_location=device)
            )

            metrics_output_path_distill = run_dir / "metrics" / "eval_distill.json"

            # Load llm_all_embeddings to pass to evaluator for ED
            data_bridge = DistillationDataBridge(cfg, teacher_outputs_dir, device)
            data_bridge.load_teacher_outputs()  # This will load llm_all_embeddings

            with time_block("eval_distill_time"):
                distill_metrics = evaluate_model(
                    student_model_distill,
                    data_module.test_dataloader(),
                    cfg,
                    device,
                    metrics_output_path_distill,
                    llm_all_emb=data_bridge.llm_all_embeddings,
                    ed_weight=cfg.distill.ed_weight,
                )
            all_results["distill"] = distill_metrics
        else:
            log.warning(
                f"Distilled model not found at {distill_model_path}. Skipping distilled evaluation."
            )

        # Save combined results
        combined_results_path = run_dir / "metrics" / "all_eval_results.json"
        with open(combined_results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        log.info(f"Combined evaluation results saved to {combined_results_path}")

    except Exception:
        log.exception("An error occurred during all evaluation run:")
        raise
    finally:
        log.info("All evaluation run finished.")


if __name__ == "__main__":
    main()
