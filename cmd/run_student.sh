#!/bin/bash

# Run the student baseline experiment
# Usage: ./cmd/run_student.sh [overrides]

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Check if running inside Docker
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container..."
    python -m src.exp.run_student_baseline experiment=student_movielens "$@"
else
    echo "Running locally..."
    # Check if poetry is available
    if command -v poetry &> /dev/null; then
        poetry run python -m src.exp.run_student_baseline experiment=student_movielens "$@"
    else
        echo "Poetry not found. Trying python directly..."
        python -m src.exp.run_student_baseline experiment=student_movielens "$@"
    fi
fi
