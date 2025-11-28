#!/bin/bash
# run_all_experiments.sh
# Runs the full pipeline for the Colab environment:
# 1. Installs dependencies
# 2. Prepares the dataset
# 3. Runs student baseline
# 4. Runs teacher model training
# 5. Runs knowledge distillation

set -eu

# Get the directory of the current script
SCRIPT_DIR=$(dirname "$0")

echo "========================================="
echo "STEP 0: Installing Poetry"
echo "========================================="
bash "$SCRIPT_DIR/00_install_poetry.sh"
echo ""

echo "========================================="
echo "STEP 1: Installing Dependencies"
echo "========================================="
bash "$SCRIPT_DIR/01_install_dependencies.sh"
echo ""

echo "========================================="
echo "STEP 2: Preparing Dataset"
echo "========================================="
bash "$SCRIPT_DIR/02_prepare_dataset.sh"
echo ""

echo "========================================="
echo "STEP 3: Running Student Baseline"
echo "========================================="
bash "$SCRIPT_DIR/10_run_student_baseline.sh"
echo ""

echo "========================================="
echo "STEP 4: Running Teacher Model Training"
echo "========================================="
bash "$SCRIPT_DIR/11_run_teacher.sh"
echo ""

echo "========================================="
echo "STEP 5: Running Knowledge Distillation"
echo "========================================="
bash "$SCRIPT_DIR/12_run_distill.sh"
echo ""

echo "========================================="
echo "All experiments complete."
echo "========================================="
