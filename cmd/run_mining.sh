#!/usr/bin/env bash

# This script runs the Active Learning (Hard Negative Mining) process.
# It executes src/utils/mine_hard_examples.py inside the 'ilora-dev-container'.

# Check if container is running
if [ ! "$(docker ps -q -f name=ilora-dev-container)" ]; then
    echo "Error: Container 'ilora-dev-container' is not running."
    echo "Please run ./env_shell/start_experiment_container.sh first."
    exit 1
fi

echo "Running Active Learning Mining..."
docker exec -it ilora-dev-container bash -c "poetry run python -m src.utils.mine_hard_examples --config-name mining $*"
