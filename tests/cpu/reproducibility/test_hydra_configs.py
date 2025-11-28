import pytest
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig

@pytest.mark.parametrize("experiment_file", [
    f for f in os.listdir(os.path.join(os.path.dirname(__file__), "../../../conf/experiment")) 
    if f.endswith(".yaml")
])
def test_experiment_configs(experiment_file):
    """
    Verify that each experiment config in conf/experiment can be loaded successfully.
    """
    config_path = "../../conf"
    experiment_name = experiment_file.replace(".yaml", "")
    
    # Check relative path validity
    if not os.path.exists(os.path.join(os.path.dirname(__file__), config_path)):
         config_path = "../../../conf"

    try:
        with hydra.initialize(version_base=None, config_path=config_path):
            cfg = hydra.compose(config_name="config", overrides=[f"experiment={experiment_name}"])
            
            assert cfg is not None
            assert isinstance(cfg, DictConfig)
            
            # Check for essential keys
            assert "dataset" in cfg
            assert "train" in cfg
            assert "teacher" in cfg
            assert "student" in cfg
            assert "distill" in cfg
            assert "eval" in cfg
            
            # Ensure defaults are populated
            assert cfg.dataset.name is not None
            assert cfg.train.batch_size is not None
            assert cfg.train.accumulate_grad_batches is not None

    except Exception as e:
        pytest.fail(f"Failed to load experiment config '{experiment_name}': {e}")
