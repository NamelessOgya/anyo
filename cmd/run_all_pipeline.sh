#!/bin/bash

set -e

echo "Starting full pipeline: Teacher -> Student Baseline -> Distillation -> Evaluation"

# Run Teacher Training
echo "--- Running Teacher Training ---"
./cmd/run_teacher_train.sh "$@"

# Run Student Baseline Training
echo "--- Running Student Baseline Training ---"
./cmd/run_student_baseline.sh "$@"

# Run Distillation Training
echo "--- Running Distillation Training ---"
./cmd/run_distill_dllm2rec.sh "$@"

# Run All Evaluation
echo "--- Running All Evaluation ---"
./cmd/run_eval_all.sh "$@"

echo "Full pipeline finished successfully."
