import subprocess
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def get_git_info() -> Dict[str, Optional[str]]:
    """
    現在のGitリポジトリの情報を取得します。
    - コミットハッシュ
    - ブランチ名
    - 変更されたファイルの有無
    - リモートURL

    Returns:
        Dict[str, Optional[str]]: Git情報を含む辞書。
                                  情報が取得できない場合はNoneが含まれます。
    """
    git_info: Dict[str, Optional[str]] = {
        "commit_hash": None,
        "branch_name": None,
        "has_uncommitted_changes": None,
        "remote_url": None,
    }

    try:
        # コミットハッシュを取得
        git_info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # ブランチ名を取得
        git_info["branch_name"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # 未コミットの変更があるか確認
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["has_uncommitted_changes"] = bool(status_output)

        # リモートURLを取得 (originが存在する場合)
        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            git_info["remote_url"] = remote_url
        except subprocess.CalledProcessError:
            git_info["remote_url"] = None # originリモートがない場合

    except subprocess.CalledProcessError as e:
        logger.warning(f"Git情報の取得中にエラーが発生しました: {e}")
        logger.warning("現在のディレクトリはGitリポジトリではないか、Gitがインストールされていません。")
        # エラーが発生した場合はすべての情報をNoneにする
        for key in git_info:
            git_info[key] = None
    except FileNotFoundError:
        logger.warning("Gitコマンドが見つかりません。Gitがインストールされていることを確認してください。")
        for key in git_info:
            git_info[key] = None

    return git_info

if __name__ == "__main__":
    # ロギング設定 (テスト用)
    logging.basicConfig(level=logging.INFO)

    info = get_git_info()
    print("--- Git Information ---")
    for key, value in info.items():
        print(f"{key}: {value}")

    # 未コミットの変更がある場合のテスト (一時ファイルを作成して確認)
    print("\n--- Testing uncommitted changes ---")
    try:
        # 一時ファイルを作成
        with open("temp_test_file.txt", "w") as f:
            f.write("test content")
        temp_info = get_git_info()
        print(f"has_uncommitted_changes (after creating temp file): {temp_info['has_uncommitted_changes']}")
    finally:
        # 一時ファイルを削除
        if os.path.exists("temp_test_file.txt"):
            os.remove("temp_test_file.txt")
        # 削除後に再度確認 (変更がないはず)
        temp_info_after_clean = get_git_info()
        print(f"has_uncommitted_changes (after cleaning temp file): {temp_info_after_clean['has_uncommitted_changes']}")
