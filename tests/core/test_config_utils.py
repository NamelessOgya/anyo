# tests/core/test_config_utils.py
import pytest
from omegaconf import OmegaConf

from src.core.config_utils import load_hydra_config

def test_load_hydra_config_defaults():
    """
    Test that load_hydra_config loads the default config correctly.
    """
    cfg = load_hydra_config(config_path="../../conf", config_name="config")

    assert cfg.dataset.name == "movielens"
    assert cfg.train.learning_rate == 1e-4
    # Check a specific value that is part of the default config
    assert cfg.student.hidden_size == 64

def test_load_hydra_config_overrides():
    """
    Test that load_hydra_config applies overrides correctly.
    """
    overrides = [
        "train.batch_size=128",
        "dataset=test_dataset_name",
        "train.learning_rate=0.0001",
    ]
    cfg = load_hydra_config(config_path="../../conf", config_name="config", overrides=overrides)

    assert cfg.train.batch_size == 128
    assert cfg.dataset.name == "test_dataset_name"
    assert cfg.train.learning_rate == 0.0001
    # Ensure other default values are still present
    assert cfg.student.hidden_size == 64 

def test_load_hydra_config_with_complex_overrides():
    """
    Test that load_hydra_config handles more complex overrides (e.g., nesting).
    """
    overrides = [
        "student.num_layers=5",
        "teacher.llm_model_name=google/gemma-2b"
    ]
    cfg = load_hydra_config(config_path="../../conf", config_name="config", overrides=overrides)

    assert cfg.student.num_layers == 5
    assert cfg.teacher.llm_model_name == "google/gemma-2b"
    assert cfg.train.batch_size == 64 # Check a default not overridden
