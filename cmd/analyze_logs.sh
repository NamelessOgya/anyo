#!/usr/bin/env bash
C=${C:-ilora-dllm2rec-dev}
echo "Targeting Container: $C"
docker exec "$C" bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python src/utils/analyze_tb_logs.py $@"
