import pytest
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def test_post_training_config_loading():
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra
    with hydra.initialize(version_base=None, config_path="../../conf"):
        # Compose config
        # We simulate the run command: experiment=bigrec_movielens
        cfg = hydra.compose(config_name="post_training_config", overrides=["experiment=bigrec_movielens"])
        
        # Verify basic structure
        # Note: 'experiment' key is not present because it uses @package _global_
        assert "post_training" in cfg
        assert "teacher" in cfg
        assert "student" in cfg
        assert "dataset" in cfg
        
        # Verify specific keys that were causing issues
        assert "popularity_path" in cfg.teacher
        assert cfg.teacher.popularity_path == "data/ml-100k/popularity_counts.pt"
        
        assert "item_embeddings_path" in cfg.teacher
        assert cfg.teacher.item_embeddings_path == "data/ml-100k/item_embeddings.pt"
        
        # Verify post_training defaults
        assert "ckpt_path" in cfg.post_training
        assert cfg.post_training.ckpt_path == "result/result_20251129_023332/checkpoints/last.ckpt"
        
        assert "sasrec_ckpt" in cfg.post_training
        
        # Verify dataset
        assert "name" in cfg.dataset
        assert cfg.dataset.name == "ml-100k"
        
def test_default_config_loading():
    # Test the main config.yaml as well just in case
    GlobalHydra.instance().clear()
    
    with hydra.initialize(version_base=None, config_path="../../conf"):
        cfg = hydra.compose(config_name="config", overrides=["experiment=ilora_movielens"])
        
        # assert "experiment" in cfg # Incorrect
        assert "teacher" in cfg
        # post_training should NOT be in default config anymore
        assert "post_training" not in cfg
