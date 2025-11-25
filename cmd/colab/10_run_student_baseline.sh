#!/bin/bash
# 10_run_student_baseline.sh
# Runs the student baseline training and evaluation.

set -eu

echo "Running student baseline..."
poetry run python -m src.exp.run_student_baseline +train=student "$@"
echo "Student baseline run complete."


