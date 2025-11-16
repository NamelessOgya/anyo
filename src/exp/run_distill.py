import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import torch

from src.core.logging import setup_logging, TensorBoardLogger, time_block
from src.core.seed import set_seed
from src.core.git_info import save_git_info
from src.core.paths import get_current_run_dir, get_data_dir, get_teacher_outputs_dir
from src.core.data_utils import preprocess_data
from src.student.datamodule import StudentDataModule
from src.distill.trainer_distill import train_distill_model
from src.student.evaluator import evaluate_model
from src.student.models import (
    get_student_model,
)  # To load the best model for evaluation
from src.distill.data_bridge import (
    DistillationDataBridge,
)  # To pass llm_all_embeddings to evaluator

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Resolve paths early
    run_dir = Path(get_current_run_dir(cfg))
    data_dir = Path(get_data_dir(cfg))
    teacher_outputs_dir = Path(get_teacher_outputs_dir(cfg))

    # Setup logging
    log_file = run_dir / "logs" / "run_distill.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file)
    log.info(f"Starting distillation run in {run_dir}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    set_seed(cfg.general.seed)

    # Save git info
    save_git_info(run_dir / "metrics" / "git_info.txt")

    # Set time_block output path
    time_block.set_output_path(run_dir / "metrics" / "time.json")

    # Prepare data (e.g., download, preprocess)
    preprocess_data(cfg, str(data_dir))

    # Initialize TensorBoard Logger
    tb_log_dir = run_dir / "tb"
    tb_logger = TensorBoardLogger(tb_log_dir)

    try:
        device = torch.device(cfg.general.device)
        data_module = StudentDataModule(cfg, str(data_dir), device)
        data_module.prepare_data()
        data_module.setup()

        # Optional: Load pre-trained student baseline model
        pre_trained_student_path = None
        if cfg.distill.get("load_baseline_model", False):
            baseline_model_path = (
                run_dir.parent
                / "baseline"
                / "models"
                / "baseline"
                / "best_baseline_model.pt"
            )
            if baseline_model_path.exists():
                pre_trained_student_path = baseline_model_path
                log.info(
                    f"Pre-trained baseline model found at {pre_trained_student_path}. Will load for distillation."
                )
            else:
                log.warning(
                    f"Configured to load baseline model, but not found at {baseline_model_path}. Starting distillation from scratch."
                )

        # Train student model with distillation
        train_distill_model(
            cfg, data_module, tb_logger, run_dir, pre_trained_student_path
        )

        # Evaluate the best distilled model
        log.info("Evaluating the best distilled model on the test set...")
        best_model_path = run_dir / "models" / "distill" / "best_distill_model.pt"

        if not best_model_path.exists():
            log.error(
                f"Best distilled model not found at {best_model_path}. Skipping evaluation."
            )
            return

        student_model = get_student_model(
            model_name=cfg.student.name,
            item_num=data_module.item_num,
            hidden_size=cfg.student.hidden_size,
            state_size=data_module.max_seq_len,
            dropout=cfg.student.dropout,
            device=device,
        ).to(device)
        student_model.load_state_dict(torch.load(best_model_path, map_location=device))

        metrics_output_path = run_dir / "metrics" / "eval_distill.json"

        # Load llm_all_embeddings to pass to evaluator for ED
        data_bridge = DistillationDataBridge(cfg, teacher_outputs_dir, device)
        data_bridge.load_teacher_outputs()  # This will load llm_all_embeddings

        evaluate_model(
            student_model,
            data_module.test_dataloader(),
            cfg,
            device,
            metrics_output_path,
            llm_all_emb=data_bridge.llm_all_embeddings,
            ed_weight=cfg.distill.ed_weight,
        )

    except Exception:
        log.exception("An error occurred during distillation run:")
        raise
    finally:
        tb_logger.close_all()
        log.info("Distillation run finished.")


if __name__ == "__main__":
    main()
