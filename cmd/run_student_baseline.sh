#!/bin/bash

set -e

if [ -f "./env_shell/env_shell/vm_hosting_sh/get_gemini.sh" ]; then
    echo "Sourcing env_shell/env_shell/vm_hosting_sh/get_gemini.sh"
    source "./env_shell/env_shell/vm_hosting_sh/get_gemini.sh"
fi

echo "Running student baseline training..."
poetry run python -m src.exp.run_student_baseline "$@"
echo "Student baseline training finished."
