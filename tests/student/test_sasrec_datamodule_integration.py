import pytest
import hydra
from omegaconf import DictConfig, OmegaConf
from src.student.datamodule import SASRecDataModule
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
import os

@pytest.fixture
def cfg():
    with hydra.initialize(version_base="1.2", config_path="../../conf"):
        cfg = hydra.compose(config_name="config", overrides=["experiment=bigrec_movielens"])
    return cfg

@patch("src.student.datamodule.pd.read_csv")
@patch("src.student.datamodule.Path.exists")
def test_sasrec_datamodule_instantiation_from_config(mock_exists, mock_read_csv, cfg):
    """
    Verifies that SASRecDataModule can be instantiated using the arguments 
    extracted from the bigrec_movielens configuration, exactly as run_bigrec.py does.
    """
    # Mock file existence to pass checks
    mock_exists.return_value = True
    
    # Mock DataFrames
    # movies.csv
    movies_df = pd.DataFrame({
        "item_id": [1, 2, 3],
        "title": ["Movie A", "Movie B", "Movie C"]
    })
    
    # train/val/test.csv
    # seq needs to be string for initial read, then processed
    data_df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "seq": ["1 2", "1 2 3", "2 3"],
        "next_item": [3, 1, 1]
    })
    
    # Configure mock_read_csv to return appropriate DFs based on call args
    def read_csv_side_effect(filepath, *args, **kwargs):
        filepath = str(filepath)
        if "movies.csv" in filepath:
            return movies_df.copy()
        else:
            return data_df.copy()
            
    mock_read_csv.side_effect = read_csv_side_effect

    # Logic from run_bigrec.py
    # ---------------------------------------------------------
    # 1. DataModule
    try:
        dm = SASRecDataModule(
            dataset_name=cfg.dataset.name,
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.teacher.batch_size,
            max_seq_len=cfg.student.max_seq_len,
            num_workers=cfg.train.num_workers,
            limit_data_rows=cfg.dataset.limit_data_rows,
            seed=cfg.seed
        )
        dm.setup()
    except TypeError as e:
        pytest.fail(f"Instantiation failed with TypeError: {e}")
    except Exception as e:
        pytest.fail(f"Instantiation failed with Exception: {e}")
        
    # ---------------------------------------------------------
    
    # Verify Attributes
    assert hasattr(dm, "mapped_id_to_title"), "SASRecDataModule missing 'mapped_id_to_title'"
    # assert hasattr(dm, "item_id_to_name"), "SASRecDataModule missing 'item_id_to_name' (alias?)"
    # Note: The code actually uses mapped_id_to_title, but let's check what exists.
    # The fix was to use mapped_id_to_title in run_bigrec.py.
    
    # Verify Config Values were passed correctly
    assert dm.batch_size == cfg.teacher.batch_size
    assert dm.max_seq_len == cfg.student.max_seq_len
    assert dm.num_workers == cfg.train.num_workers
    
    print("\nSuccessfully instantiated SASRecDataModule from config!")
    print(f"mapped_id_to_title keys: {list(dm.mapped_id_to_title.keys())}")
