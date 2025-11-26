#!/bin/bash

docker exec ilora-dev-container bash -c "PYTHONPATH=/workspace poetry run python -m src.core.preprocess_data --data_dir data/ml-1m"
