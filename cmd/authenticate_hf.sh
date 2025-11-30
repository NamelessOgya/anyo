#!/usr/bin/env bash

# This script handles Hugging Face authentication.
# It runs inside the 'ilora-dllm2rec-dev'.

echo "Starting Hugging Face Authentication..."
echo "You will be asked to enter your Hugging Face Access Token."

C=${C:-ilora-dllm2rec-dev}
echo "Targeting Container: $C"

docker exec -it "$C" bash -c "poetry run python authenticate_hf.py"

echo "Done."
