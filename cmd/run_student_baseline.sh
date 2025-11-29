#!/usr/bin/env bash
docker exec ilora-dllm2rec-dev bash -c "PYTHONPATH=/workspace poetry run python -m src.exp.run_student_baseline train=student $@"

