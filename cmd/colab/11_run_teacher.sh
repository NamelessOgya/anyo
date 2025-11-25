#!/bin/bash
# 11_run_teacher.sh
# Runs the teacher model training and evaluation.
# It requires the path to the student model checkpoint as the first argument.

set -eu

# Add poetry to the PATH
export PATH="/root/.local/bin:$PATH"

STUDENT_CHECKPOINT_PATH="$1"
shift # Remove the first argument (STUDENT_CHECKPOINT_PATH) from the list of arguments

if [ -z "$STUDENT_CHECKPOINT_PATH" ]; then
    echo "Error: Student checkpoint path not provided."
    echo "Usage: $0 <path_to_student_checkpoint> [hydra_args...]"
    exit 1
fi

echo "Using student checkpoint: $STUDENT_CHECKPOINT_PATH"

echo "Running teacher model training..."
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" CUDA_LAUNCH_BLOCKING=1 poetry run python -m src.exp.run_teacher +train=teacher "teacher.rec_model_checkpoint_path='$STUDENT_CHECKPOINT_PATH'" "$@"
echo "Teacher model run complete."



