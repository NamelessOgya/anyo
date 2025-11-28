#!/bin/bash
# 02_prepare_dataset.sh
# Prepares the dataset by calling the existing download script.

set -eu

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$0")
# Get the project root directory (assuming the script is in cmd/colab/)
PROJECT_ROOT="$SCRIPT_DIR/../.."

# Path to the original download script
DOWNLOAD_SCRIPT="$PROJECT_ROOT/data/download_movielens.sh"

if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "Error: Download script not found at $DOWNLOAD_SCRIPT"
    exit 1
fi

echo "Preparing dataset..."
bash "$DOWNLOAD_SCRIPT"

echo "Preprocessing data (splitting into train/val/test)..."
poetry run python -m src.core.preprocess_data --data_dir "$PROJECT_ROOT/data/ml-1m"

echo "Dataset preparation and preprocessing complete."

