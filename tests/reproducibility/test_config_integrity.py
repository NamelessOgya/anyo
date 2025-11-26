
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
import os

def test_default_config_has_no_batch_size():
    # Manually load default.yaml to check raw content
    # We can't easily use hydra compose for a single file if it's not a full config group, 
    # but we can check the file content or try to compose a minimal config.
    # Simpler: just check if 'batch_size' key exists in the loaded yaml.
    
    # Using OmegaConf to load the file directly
    default_conf_path = "conf/train/default.yaml"
    conf = OmegaConf.load(default_conf_path)
    
    assert "batch_size" not in conf, "conf/train/default.yaml should not contain 'batch_size'"

def test_experiment_configs_batch_size():
    # Initialize hydra
    # We need to be careful about initialization if it's already initialized in the same process.
    # pytest-hydra or clearing the instance might be needed.
    # For simplicity, we use the context manager.
    
    config_path = "../../conf"
    
    # 1. Test Teacher Config
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config", overrides=["experiment=ilora_teacher"])
        assert cfg.train.batch_size == 16, f"Teacher batch_size should be 16, got {cfg.train.batch_size}"
        assert cfg.train.accumulate_grad_batches == 32

    # 2. Test Student Config
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config", overrides=["experiment=ilora_student"])
        assert cfg.train.batch_size == 256, f"Student batch_size should be 256, got {cfg.train.batch_size}"

    # 3. Test Distill Config
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config", overrides=["experiment=ilora_distill"])
        assert cfg.train.batch_size == 32, f"Distill batch_size should be 32, got {cfg.train.batch_size}"

    # 4. Test MovieLens Experiment (which we modified to use teacher)
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config", overrides=["experiment=ilora_movielens"])
        assert cfg.train.batch_size == 16, f"MovieLens experiment should use teacher config (16), got {cfg.train.batch_size}"
