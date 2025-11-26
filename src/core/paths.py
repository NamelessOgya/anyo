import os
from pathlib import Path

def get_project_root() -> Path:
    """
    プロジェクトのルートディレクトリのパスを返します。
    このファイル (`src/core/paths.py`) から見て、`v2` ディレクトリがルートと仮定します。
    """
    return Path(__file__).parent.parent.parent

def get_data_dir() -> Path:
    """
    データディレクトリのパスを返します。
    """
    return get_project_root() / "data"

def get_result_dir() -> Path:
    """
    実験結果ディレクトリのパスを返します。
    """
    return get_project_root() / "result"

def get_conf_dir() -> Path:
    """
    設定ファイルディレクトリのパスを返します。
    """
    return get_project_root() / "conf"

def get_src_dir() -> Path:
    """
    ソースコードディレクトリのパスを返します。
    """
    return get_project_root() / "src"

def get_env_shell_dir() -> Path:
    """
    環境シェルスクリプトディレクトリのパスを返します。
    """
    return get_project_root() / "env_shell"

if __name__ == "__main__":
    print(f"Project Root: {get_project_root()}")
    print(f"Data Dir: {get_data_dir()}")
    print(f"Result Dir: {get_result_dir()}")
    print(f"Conf Dir: {get_conf_dir()}")
    print(f"Src Dir: {get_src_dir()}")
    print(f"Env Shell Dir: {get_env_shell_dir()}")
