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
from src.teacher.factory import get_teacher_model

log = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set default dtype
    torch.set_default_dtype(torch.float32)

    # Resolve paths early
    run_dir = Path(get_current_run_dir(cfg))
    data_dir = Path(get_data_dir(cfg))
    teacher_outputs_dir = Path(get_teacher_outputs_dir(cfg))

    # Setup logging
    log_file = run_dir / "logs" / "run_teacher.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file)
    log.info(f"Starting teacher run in {run_dir}")
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
        teacher_model = get_teacher_model(cfg)

        # Train teacher model
        teacher_model.train(cfg, tb_logger, run_dir)

        # Export teacher outputs for DLLM2Rec
        teacher_model.export_for_dllm2rec(cfg, run_dir, data_dir, teacher_outputs_dir)

    except Exception:
        log.exception("An error occurred during teacher run:")
        raise
    finally:
        tb_logger.close_all()
        log.info("Teacher run finished.")


if __name__ == "__main__":
    main()
