from abc import ABC, abstractmethod
from omegaconf import DictConfig
import logging
from pathlib import Path

# Assuming these are available from core.logging
from src.core.logging import TensorBoardLogger

log = logging.getLogger(__name__)


class ITeacherRecommender(ABC):
    """
    Abstract base class for a teacher recommender model.
    Defines the interface for training the teacher model.
    """

    @abstractmethod
    def train(self, cfg: DictConfig, tb_logger: TensorBoardLogger, run_dir: Path):
        """
        Trains the teacher recommender model.
        :param cfg: Hydra configuration object.
        :param tb_logger: TensorBoardLogger instance.
        :param run_dir: Path to the current experiment run directory.
        """
        pass


class ITeacherExporter(ABC):
    """
    Abstract base class for exporting teacher model outputs for distillation.
    """

    @abstractmethod
    def export_for_dllm2rec(
        self, cfg: DictConfig, run_dir: Path, data_dir: Path, teacher_outputs_dir: Path
    ):
        """
        Exports necessary files (embeddings, rankings, confidences) for DLLM2Rec distillation.
        :param cfg: Hydra configuration object.
        :param run_dir: Path to the current experiment run directory.
        :param data_dir: Path to the processed data directory.
        :param teacher_outputs_dir: Path where teacher outputs should be saved.
        """
        pass


class ITeacherModel(ITeacherRecommender, ITeacherExporter, ABC):
    """
    Combines both training and exporting interfaces for a complete teacher model.
    """

    pass


if __name__ == "__main__":
    # Example usage and demonstration of interfaces
    from omegaconf import OmegaConf
    from src.core.logging import setup_logging
    import shutil

    # Setup logging
    setup_logging(Path("temp_log.log"))

    # Dummy config
    dummy_cfg = OmegaConf.create(
        {
            "teacher": {
                "name": "ilora",
                "llm_path": "/path/to/Llama-2-7b-hf",
                "batch_size": 64,
                "lr": 1e-4,
                "num_epochs": 1,
                "prompt_template": "default",
                "save_interval": 1,
            },
            "dataset": {"name": "movielens"},
        }
    )

    class DummyTeacher(ITeacherModel):
        def train(self, cfg: DictConfig, tb_logger: TensorBoardLogger, run_dir: Path):
            log.info(f"DummyTeacher: Training with config: {cfg.teacher.name}")
            log.info(f"DummyTeacher: Using run_dir: {run_dir}")
            writer = tb_logger.get_writer("dummy_teacher")
            writer.add_scalar("dummy/loss", 0.1, 0)
            writer.close()

        def export_for_dllm2rec(
            self,
            cfg: DictConfig,
            run_dir: Path,
            data_dir: Path,
            teacher_outputs_dir: Path,
        ):
            log.info(
                f"DummyTeacher: Exporting for DLLM2Rec for dataset: {cfg.dataset.name}"
            )
            log.info(f"DummyTeacher: Saving outputs to: {teacher_outputs_dir}")
            teacher_outputs_dir.mkdir(parents=True, exist_ok=True)
            (teacher_outputs_dir / "all_embeddings.pt").touch()
            (teacher_outputs_dir / "myrank_train.txt").touch()
            (teacher_outputs_dir / "confidence_train.txt").touch()
            log.info("Dummy teacher outputs created.")

    log.info("Instantiating DummyTeacher...")
    dummy_teacher = DummyTeacher()

    temp_run_dir = Path("temp_run_teacher")
    temp_data_dir = Path("temp_data_teacher")
    temp_teacher_outputs_dir = (
        temp_data_dir / "teacher_outputs" / dummy_cfg.dataset.name
    )

    tb_logger = TensorBoardLogger(temp_run_dir / "tb")

    log.info("\nCalling train method...")
    dummy_teacher.train(dummy_cfg, tb_logger, temp_run_dir)

    log.info("\nCalling export_for_dllm2rec method...")
    dummy_teacher.export_for_dllm2rec(
        dummy_cfg, temp_run_dir, temp_data_dir, temp_teacher_outputs_dir
    )

    tb_logger.close_all()

    # Clean up
    shutil.rmtree(temp_run_dir, ignore_errors=True)
    shutil.rmtree(temp_data_dir, ignore_errors=True)
    Path("temp_log.log").unlink(missing_ok=True)
    log.info("\nCleaned up temporary directories and log file.")
