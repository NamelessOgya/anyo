#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Source environment setup scripts if they exist
# Assuming env_shell/vm_hosting_sh/additional_installs.sh and others might be needed
# For now, we'll just source a generic one if it exists.
# if [ -f "./env_shell/env_shell/vm_hosting_sh/get_gemini.sh" ]; then
#     echo "Sourcing env_shell/env_shell/vm_hosting_sh/get_gemini.sh"
#     source "./env_shell/env_shell/vm_hosting_sh/get_gemini.sh"
# fi

echo "Running teacher model training..."
poetry run python -m src.exp.run_teacher "$@"
echo "Teacher model training finished."
