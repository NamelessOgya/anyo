import logging
import sys
from pathlib import Path

def setup_logging(log_dir: Path = None, level=logging.INFO):
    """
    ロギングを設定します。
    コンソール出力と、指定されたディレクトリへのファイル出力を設定します。

    Args:
        log_dir (Path, optional): ログファイルを保存するディレクトリ。指定しない場合はファイル出力なし。
        level (int, optional): ロギングレベル。デフォルトはlogging.INFO。
    """
    # ルートロガーを取得
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 既存のハンドラをすべて削除
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # フォーマッタの定義
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ファイルハンドラ
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "experiment.log")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.info("Logging setup complete.")

if __name__ == "__main__":
    from src.core.paths import get_project_root

    # ログディレクトリを指定して設定
    log_output_dir = get_project_root() / "logs"
    setup_logging(log_dir=log_output_dir, level=logging.DEBUG)

    logging.debug("これはデバッグメッセージです。")
    logging.info("これは情報メッセージです。")
    logging.warning("これは警告メッセージです。")
    logging.error("これはエラーメッセージです。")
    logging.critical("これは致命的なエラーメッセージです。")

    # ログディレクトリを指定しない場合
    print("\n--- ファイル出力なしのロギング ---")
    setup_logging(level=logging.INFO)
    logging.info("ファイル出力なしでロギング中。")
