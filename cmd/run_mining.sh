#!/usr/bin/env bash

# This script runs the Active Learning (Hard Negative Mining) process.
# It executes src/utils/mine_hard_examples.py inside the 'ilora-dllm2rec-dev'.

# Check if container is running
if [ ! "$(docker ps -q -f name=ilora-dllm2rec-dev)" ]; then
    echo "Error: Container 'ilora-dllm2rec-dev' is not running."
    echo "Please run ./env_shell/start_experiment_container.sh first."
    exit 1
fi

echo "Running Active Learning Mining..."
docker exec -it ilora-dllm2rec-dev bash -c "poetry run python -m src.utils.mine_hard_examples --config-name mining $*"
