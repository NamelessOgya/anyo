#!/usr/bin/env bash
CONTAINER_NAME=${CONTAINER_NAME:-ilora-dllm2rec-dev}
echo "Targeting Container: $CONTAINER_NAME"
docker exec "$CONTAINER_NAME" bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python -m src.exp.run_teacher $@"
