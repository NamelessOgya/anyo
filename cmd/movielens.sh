#!/bin/bash
# cmd/movielens.sh
# MovieLens 1M データセットをダウンロードして data/ml-1m に配置するスクリプト

set -eu

# プロジェクトルートディレクトリに移動 (スクリプトの場所から推定)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
ML_1M_DIR="${DATA_DIR}/ml-1m"

# dataディレクトリの作成
mkdir -p "${DATA_DIR}"

echo "Downloading MovieLens 1M dataset..."
# ダウンロード (curlを使用、なければwget)
if command -v curl >/dev/null 2>&1; then
    curl -fL -o "${DATA_DIR}/ml-1m.zip" "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${DATA_DIR}/ml-1m.zip" "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
else
    echo "Error: curl or wget is required."
    exit 1
fi

echo "Extracting dataset..."
# 解凍
# 解凍
if command -v unzip >/dev/null 2>&1; then
    unzip -o "${DATA_DIR}/ml-1m.zip" -d "${DATA_DIR}"
else
    echo "'unzip' command not found. Falling back to python..."
    python3 -c "import zipfile; import sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "${DATA_DIR}/ml-1m.zip" "${DATA_DIR}"
fi

# 不要なzipファイルを削除
rm "${DATA_DIR}/ml-1m.zip"

echo "MovieLens 1M dataset is ready at ${ML_1M_DIR}"
