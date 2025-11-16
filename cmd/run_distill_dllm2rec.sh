#!/bin/bash

set -e

if [ -f "./env_shell/env_shell/vm_hosting_sh/get_gemini.sh" ]; then
    echo "Sourcing env_shell/env_shell/vm_hosting_sh/get_gemini.sh"
    source "./env_shell/env_shell/vm_hosting_sh/get_gemini.sh"
fi

echo "Running distillation training..."
poetry run python -m src.exp.run_distill "$@"
echo "Distillation training finished."
