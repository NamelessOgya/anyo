#!/bin/bash
# Run within container: ./cmd/tune_popularity.sh [optional overrides]

# Default to ilora-dllm2rec-dev container if not specified
CONTAINER_NAME="${C:-ilora-dllm2rec-dev}"

echo "Running tune_popularity.py in container: $CONTAINER_NAME"

# Example usage: ./cmd/tune_popularity.sh experiment=moe_bigrec_movielens
docker exec -it "$CONTAINER_NAME" /bin/bash -c "export PYTHONPATH=. && python src/exp/tune_popularity.py $@"
