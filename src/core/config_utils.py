# src/core/config_utils.py
import sys
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

def load_hydra_config(
    config_path: str = "../conf",
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    HydraをJupyter NotebookやプレーンなPythonスクリプト環境で初期化し、
    設定をロードするための共通関数。
    @hydra.mainの代替として機能する。

    Args:
        config_path (str): このファイルから見たHydra設定ファイルの相対パス。
        config_name (str): メインとなる設定ファイル名（.yaml拡張子は除く）。
        overrides (Optional[List[str]]): コマンドラインからの上書き引数のリスト。
                                         例: ["train.batch_size=64", "dataset=movielens"]

    Returns:
        DictConfig: ロードされたHydraの設定オブジェクト。
    """
    if overrides is None:
        overrides = []

    # sys.argvからJupyterの引数などをフィルタリング
    original_argv = sys.argv.copy()
    # poetry run python -m ... のような実行に対応するため、最初の引数のみを使う
    sys.argv = [original_argv[0]] 

    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(
            config_path=config_path, job_name="anyo_refactor_app", version_base=None
        )
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

        # OmegaConf.to_yaml(cfg) などを利用して、ロードされた設定を確認できる
        # print(OmegaConf.to_yaml(cfg))

    finally:
        # sys.argvを元に戻す
        sys.argv = original_argv

    return cfg
