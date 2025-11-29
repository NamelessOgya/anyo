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
# Run python script using poetry
PYTHONPATH=. poetry run python src/core/preprocess_data.py --data_dir "$DATA_DIR"

echo "Done!"
