
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from omegaconf import OmegaConf
from src.exp.run_teacher import upload_results

@patch("src.exp.run_teacher.shutil.copytree")
def test_upload_results_enabled(mock_copytree):
    # Setup
    cfg = OmegaConf.create({"upload_path": "/fake/upload/path"})
    output_dir = Path("/fake/output/dir")
    
    # Execute
    upload_results(cfg, output_dir)
    
    # Verify
    mock_copytree.assert_called_once()
    args, kwargs = mock_copytree.call_args
    assert args[0] == output_dir
    assert args[1] == Path("/fake/upload/path")
    assert kwargs.get("dirs_exist_ok") is True

@patch("src.exp.run_teacher.shutil.copytree")
def test_upload_results_disabled(mock_copytree):
    # Setup
    cfg = OmegaConf.create({"upload_path": None})
    output_dir = Path("/fake/output/dir")
    
    # Execute
    upload_results(cfg, output_dir)
    
    # Verify
    mock_copytree.assert_not_called()

@patch("src.exp.run_teacher.shutil.copytree")
def test_upload_results_exception(mock_copytree):
    # Setup
    cfg = OmegaConf.create({"upload_path": "/fake/upload/path"})
    output_dir = Path("/fake/output/dir")
    
    # Simulate exception
    mock_copytree.side_effect = Exception("Copy failed")
    
    # Execute (should not raise, just log error)
    try:
        upload_results(cfg, output_dir)
    except Exception:
        pytest.fail("upload_results raised exception instead of catching it")
    
    # Verify called
    mock_copytree.assert_called_once()
