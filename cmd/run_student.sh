#!/usr/bin/env bash
docker exec ilora-dllm2rec-dev bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python -m src.exp.run_student_baseline experiment=student_movielens $@"
