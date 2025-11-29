#!/usr/bin/env bash

# This script handles Hugging Face authentication.
# It runs inside the 'ilora-dllm2rec-dev'.

echo "Starting Hugging Face Authentication..."
echo "You will be asked to enter your Hugging Face Access Token."

docker exec -it ilora-dllm2rec-dev bash -c "poetry run python authenticate_hf.py"

echo "Done."
