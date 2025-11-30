#!/usr/bin/env bash

# This script runs the Active Learning (Hard Negative Mining) process.
# It executes src/utils/mine_hard_examples.py inside the container.

CONTAINER_NAME=${CONTAINER_NAME:-ilora-dllm2rec-dev}

# Check if container is running
if [ ! "$(docker ps -q -f name="$CONTAINER_NAME")" ]; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    echo "Please run ./env_shell/start_experiment_container.sh first."
    exit 1
fi

echo "Running Active Learning Mining in $CONTAINER_NAME..."
docker exec "$CONTAINER_NAME" bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python -m src.utils.mine_hard_examples --config-name mining $*"
