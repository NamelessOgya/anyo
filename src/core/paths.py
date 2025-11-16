import os
from omegaconf import DictConfig


def get_data_dir(cfg: DictConfig) -> str:
    """Calculates the absolute path to the data directory."""
    return os.path.abspath(cfg.paths.data_dir)


def get_result_root(cfg: DictConfig) -> str:
    """Calculates the absolute path to the result root directory."""
    return os.path.abspath(cfg.paths.result_root)


def get_teacher_outputs_dir(cfg: DictConfig) -> str:
    """Calculates the absolute path to the teacher outputs directory."""
    # Hydra's interpolation should handle this, but we ensure it's absolute
    return os.path.abspath(cfg.paths.teacher_outputs_dir)


def get_current_run_dir(cfg: DictConfig) -> str:
    """Returns the absolute path to the current Hydra run directory."""
    # Hydra automatically sets the working directory to the run directory if chdir is true
    # So, '.' refers to the run directory.
    return os.path.abspath(".")
