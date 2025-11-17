#!/usr/bin/env bash
docker exec ilora-dev-container bash -c "PYTHONPATH=/workspace poetry run python -m src.exp.run_distill $@"
