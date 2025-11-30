#!/bin/bash
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."

DATA_DIR="data/ml-100k"
ZIP_FILE="ml-100k.zip"
URL="https://files.grouplens.org/datasets/movielens/ml-100k.zip"

echo "Checking for MovieLens-100k data..."

if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory $DATA_DIR not found. Downloading..."
    
    # Create data directory if it doesn't exist
    mkdir -p data
    
    # Download zip if not present
    if [ ! -f "$ZIP_FILE" ]; then
        echo "Downloading $ZIP_FILE..."
        curl -O "$URL"
    fi
    
    # Unzip
    echo "Unzipping..."
    unzip -o "$ZIP_FILE" -d data/
    
    # Cleanup zip (optional, keeping it for now as cache)
    # rm "$ZIP_FILE"
    
    echo "Download and extraction complete."
else
    echo "Data directory $DATA_DIR exists. Skipping download."
fi

echo "Running preprocessing..."

if command -v poetry &> /dev/null; then
    # Poetry found, run locally (or inside container if poetry is installed there)
    PYTHONPATH=. poetry run python src/core/preprocess_data.py --data_dir "$DATA_DIR" --dataset_type ml-100k
else
    # Poetry not found, assume we are on host and try to run via Docker
        C=${C:-ilora-dllm2rec-dev}
        echo "Poetry not found. Attempting to run inside Docker container '$C'..."
        if docker ps | grep -q "$C"; then
            docker exec "$C" bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python -m src.data.preprocess_data --data_dir $DATA_DIR --dataset_type ml-100k"
        else
            echo "Error: Poetry not found and Docker container '$C' is not running."
        exit 1
    fi
fi

echo "Done!"
