#!/bin/bash
# 12_run_distill.sh
# Runs the knowledge distillation process.

set -eu

echo "Running knowledge distillation..."
poetry run python -m src.exp.run_distill train.batch_size=64
echo "Knowledge distillation run complete."
