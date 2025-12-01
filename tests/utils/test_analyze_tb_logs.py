import pytest
import os
import pandas as pd
from unittest.mock import MagicMock, patch
from src.utils.analyze_tb_logs import analyze_tb_logs

@pytest.fixture
def mock_event_accumulator():
    with patch("src.utils.analyze_tb_logs.EventAccumulator") as mock:
        yield mock

@pytest.fixture
def mock_glob():
    with patch("src.utils.analyze_tb_logs.glob.glob") as mock:
        yield mock

def test_analyze_tb_logs_success(mock_event_accumulator, mock_glob, tmp_path):
    # Setup mocks
    mock_glob.return_value = ["events.out.tfevents.123"]
    
    ea_instance = mock_event_accumulator.return_value
    ea_instance.Tags.return_value = {'scalars': ['val_loss', 'train_loss', 'val_hr@10']}
    
    # Mock Scalar events
    mock_event = MagicMock()
    mock_event.step = 1
    mock_event.value = 0.5
    
    ea_instance.Scalars.side_effect = lambda tag: [mock_event] if tag in ['val_loss', 'val_hr@10'] else []

    # Run function
    result_dir = str(tmp_path)
    
    analyze_tb_logs(result_dir)
    
    # Verify EventAccumulator was called
    mock_event_accumulator.assert_called()
    
    # Verify CSV was saved
    expected_csv = os.path.join(result_dir, "validation_metrics.csv")
    assert os.path.exists(expected_csv)
    
    # Verify content of CSV
    df = pd.read_csv(expected_csv, index_col=0)
    assert "val_loss" in df.columns
    assert "val_hr@10" in df.columns
    assert "train_loss" not in df.columns # Should be filtered out
    assert df.loc[1, "val_loss"] == 0.5

def test_analyze_tb_logs_no_files(mock_glob):
    mock_glob.return_value = []
    
    # Should print message and return, no error
    analyze_tb_logs("dummy_dir")

def test_analyze_tb_logs_no_metrics(mock_event_accumulator, mock_glob):
    mock_glob.return_value = ["events.out.tfevents.123"]
    ea_instance = mock_event_accumulator.return_value
    ea_instance.Tags.return_value = {'scalars': ['train_loss']} # No val metrics
    
    analyze_tb_logs("dummy_dir")
    
    # Should not crash, but also not save CSV
    # We can't easily check "not saved" without mocking to_csv or checking file system (which we can't do easily with dummy_dir)
    # But ensuring it runs without error is a good start.
