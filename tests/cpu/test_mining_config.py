import pytest
import hydra
from omegaconf import DictConfig, OmegaConf
from src.core.paths import get_project_root
import os
from unittest.mock import patch, MagicMock
import torch

# Test configuration loading
def test_mining_config_structure():
    """
    Verify that conf/mining.yaml can be loaded and contains expected keys.
    This catches issues like 'Key not in struct' for critical parameters.
    """
    with hydra.initialize(version_base="1.2", config_path="../../conf"):
        cfg = hydra.compose(config_name="mining", overrides=["student_checkpoint_path=dummy.ckpt"])
        
        # Check Active Learning config
        assert "active_learning" in cfg
        assert "strategy_name" in cfg.active_learning
        assert "mining_ratio" in cfg.active_learning
        assert "hard_indices_output_path" in cfg.active_learning
        
        # Check Train config (should be student)
        assert "train" in cfg
        assert "batch_size" in cfg.train
        
        # Check Student/Model config
        # The script uses cfg.student for model params
        assert "student" in cfg
        assert "max_seq_len" in cfg.student
        assert "num_candidates" in cfg.student # This was the missing key
        assert "hidden_size" in cfg.student

# Test script execution logic (mocked)
@patch("src.utils.mine_hard_examples.SASRecDataModule")
@patch("src.utils.mine_hard_examples.SASRecTrainer")
@patch("src.utils.mine_hard_examples.SASRec")
@patch("src.utils.mine_hard_examples.torch.load")
@patch("src.utils.mine_hard_examples.torch.save")
@patch("src.utils.active_learning.get_strategy")
def test_mine_hard_examples_script(mock_get_strategy, mock_save, mock_load, mock_sasrec, mock_trainer, mock_dm):
    """
    Verify that src.utils.mine_hard_examples.main runs without config errors.
    We mock the heavy lifting (DataModule, Model, Trainer) to focus on config usage.
    """
    from src.utils.mine_hard_examples import main
    
    # Setup mocks
    mock_dm_instance = mock_dm.return_value
    mock_dm_instance.num_items = 100
    mock_dm_instance.train_dataset = MagicMock()
    
    mock_trainer_instance = mock_trainer.load_from_checkpoint.return_value
    mock_model = mock_trainer_instance.model
    
    # Mock strategy
    mock_strategy_instance = mock_get_strategy.return_value
    mock_strategy_instance.select_indices.return_value = [1, 2, 3]
    
    # Load config
    with hydra.initialize(version_base="1.2", config_path="../../conf"):
        cfg = hydra.compose(config_name="mining", overrides=["student_checkpoint_path=dummy.ckpt"])
        
        # Run main
        # We need to ensure os.path.exists returns True for the dummy checkpoint
        with patch("os.path.exists", return_value=True):
            main(cfg)
            
    # Verify calls
    # Ensure DataModule was initialized with correct config values
    mock_dm.assert_called_once()
    call_kwargs = mock_dm.call_args.kwargs
    assert call_kwargs["batch_size"] == cfg.train.batch_size
    assert call_kwargs["max_seq_len"] == cfg.student.max_seq_len
    assert call_kwargs["num_candidates"] == cfg.student.num_candidates
    
    # Ensure Strategy was initialized
    mock_get_strategy.assert_called_once()
    args, _ = mock_get_strategy.call_args
    assert args[0] == cfg.active_learning.strategy_name
    assert args[3] == cfg.active_learning.mining_ratio
    
    # Ensure save was called
    mock_save.assert_called_once()
