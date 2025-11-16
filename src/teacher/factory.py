from omegaconf import DictConfig
from src.teacher.interfaces import ITeacherModel
from src.teacher.ilora_backend import ILoRABackend
import logging

log = logging.getLogger(__name__)


def get_teacher_model(cfg: DictConfig) -> ITeacherModel:
    """
    Factory function to create and return a teacher model instance
    based on the configuration.
    """
    teacher_name = cfg.teacher.name
    if teacher_name == "ilora":
        return ILoRABackend()
    else:
        raise ValueError(f"Unknown teacher model: {teacher_name}")


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    from src.core.logging import setup_logging
    from pathlib import Path

    # Setup logging
    setup_logging(Path("temp_log_factory.log"))

    # Dummy config
    dummy_cfg = OmegaConf.create(
        {
            "teacher": {
                "name": "ilora",
                "llm_path": "/path/to/Llama-2-7b-hf",
                "batch_size": 64,
                "lr": 1e-4,
                "num_epochs": 5,
                "prompt_template": "default",
                "save_interval": 1,
            }
        }
    )

    log.info("Getting iLoRA teacher model...")
    teacher_model = get_teacher_model(dummy_cfg)
    log.info(f"Successfully created teacher model of type: {type(teacher_model)}")

    # Test with an unknown teacher
    dummy_cfg.teacher.name = "unknown_teacher"
    try:
        get_teacher_model(dummy_cfg)
    except ValueError as e:
        log.info(f"Caught expected error for unknown teacher: {e}")

    # Clean up
    Path("temp_log_factory.log").unlink(missing_ok=True)
