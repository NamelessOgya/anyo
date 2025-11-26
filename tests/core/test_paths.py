import pytest
from pathlib import Path
from src.core import paths

def test_get_project_root():
    """
    get_project_root() が正しいプロジェクトルートを返すかテストします。
    """
    project_root = paths.get_project_root()
    
    # パスがPathオブジェクトであることを確認
    assert isinstance(project_root, Path)
    
    # パスが存在し、ディレクトリであることを確認
    assert project_root.exists()
    assert project_root.is_dir()
    
    # プロジェクトルートに 'src' や 'pyproject.toml' が存在することを確認
    assert (project_root / "src").is_dir()
    assert (project_root / "pyproject.toml").is_file()

def test_get_data_dir():
    """
    get_data_dir() が正しいデータディレクトリを返すかテストします。
    """
    data_dir = paths.get_data_dir()
    project_root = paths.get_project_root()
    
    assert data_dir == project_root / "data"
    assert data_dir.exists()
    assert data_dir.is_dir()

def test_get_result_dir():
    """
    get_result_dir() が正しい結果ディレクトリを返すかテストします。
    """
    result_dir = paths.get_result_dir()
    project_root = paths.get_project_root()
    
    assert result_dir == project_root / "result"
    assert result_dir.exists()
    assert result_dir.is_dir()

def test_get_conf_dir():
    """
    get_conf_dir() が正しい設定ディレクトリを返すかテストします。
    """
    conf_dir = paths.get_conf_dir()
    project_root = paths.get_project_root()
    
    assert conf_dir == project_root / "conf"
    assert conf_dir.exists()
    assert conf_dir.is_dir()

def test_get_src_dir():
    """
    get_src_dir() が正しいソースディレクトリを返すかテストします。
    """
    src_dir = paths.get_src_dir()
    project_root = paths.get_project_root()
    
    assert src_dir == project_root / "src"
    assert src_dir.exists()
    assert src_dir.is_dir()
