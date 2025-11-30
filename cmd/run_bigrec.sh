#!/usr/bin/env bash
C=${C:-ilora-dllm2rec-dev}
docker exec "$C" bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python -m src.exp.run_bigrec experiment=bigrec_movielens $@"
