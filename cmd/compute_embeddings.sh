#!/bin/bash
# Run within container: ./cmd/compute_embeddings.sh

# Default to ilora-dllm2rec-dev container if not specified
CONTAINER_NAME="${C:-ilora-dllm2rec-dev}"

echo "Running compute_embeddings.py in container: $CONTAINER_NAME"

docker exec -it "$CONTAINER_NAME" /bin/bash -c "export PYTHONPATH=. && python src/exp/compute_embeddings.py experiment=moe_bigrec_movielens"
