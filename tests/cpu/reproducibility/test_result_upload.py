
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from omegaconf import OmegaConf
from src.exp.run_teacher import upload_results
import subprocess

@patch("src.exp.run_teacher.subprocess.run")
@patch("src.exp.run_teacher.Path.mkdir")
def test_upload_results_enabled(mock_mkdir, mock_subprocess):
    # Setup
    cfg = OmegaConf.create({"upload_path": "/fake/upload/path"})
    output_dir = Path("/fake/output/dir")
    
    # Execute
    upload_results(cfg, output_dir)
    
    # Verify
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_subprocess.assert_called_once()
    args, kwargs = mock_subprocess.call_args
    assert "cp -R" in args[0]
    assert "/fake/output/dir/*" in args[0]
    assert "/fake/upload/path/" in args[0]
    assert kwargs.get("shell") is True

@patch("src.exp.run_teacher.subprocess.run")
def test_upload_results_disabled(mock_subprocess):
    # Setup
    cfg = OmegaConf.create({"upload_path": None})
    output_dir = Path("/fake/output/dir")
    
    # Execute
    upload_results(cfg, output_dir)
    
    # Verify
    mock_subprocess.assert_not_called()

@patch("src.exp.run_teacher.subprocess.run")
@patch("src.exp.run_teacher.Path.mkdir")
def test_upload_results_exception(mock_mkdir, mock_subprocess):
    # Setup
    cfg = OmegaConf.create({"upload_path": "/fake/upload/path"})
    output_dir = Path("/fake/output/dir")
    
    # Simulate exception
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "cp", stderr="Copy failed")
    
    # Execute (should not raise, just log error)
    try:
        upload_results(cfg, output_dir)
    except Exception:
        pytest.fail("upload_results raised exception instead of catching it")
    
    # Verify called
    mock_subprocess.assert_called_once()
