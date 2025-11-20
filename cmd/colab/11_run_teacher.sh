#!/bin/bash
# 11_run_teacher.sh
# Runs the teacher model training and evaluation.
# It automatically finds the latest student model checkpoint.

set -eu

echo "Finding latest student model checkpoint..."

# Find the latest result directory from a student baseline run
# We identify student runs by the directory prefix "student_baseline_"
LATEST_STUDENT_DIR=$(find result -type d -name "student_baseline_*" | sort -r | head -n 1)

if [ -z "$LATEST_STUDENT_DIR" ]; then
    echo "Error: No student baseline results found in the 'result/' directory."
    echo "Please run the student baseline first (./cmd/colab/10_run_student_baseline.sh)."
    exit 1
fi
echo "Found latest student run directory: $LATEST_STUDENT_DIR"

# Find the checkpoint file in that directory
# We look for .ckpt files, assuming there's only one, or taking the first one
CHECKPOINT_FILE=$(find "$LATEST_STUDENT_DIR" -name "*.ckpt" | head -n 1)

if [ -z "$CHECKPOINT_FILE" ]; then
    echo "Error: No .ckpt file found in $LATEST_STUDENT_DIR"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_FILE"

echo "Running teacher model training..."
# Override the hydra config with the found checkpoint path AND a smaller batch size
poetry run python -m src.exp.run_teacher "teacher.rec_model_checkpoint_path='$CHECKPOINT_FILE'" train.batch_size=16
echo "Teacher model run complete."



