#!/usr/bin/env bash
docker exec ilora-dev-container bash -c "PYTHONPATH=/workspace /opt/conda/bin/poetry run python -m src.exp.run_teacher $@"
