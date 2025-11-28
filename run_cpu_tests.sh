#!/bin/bash
# Run all tests in tests/cpu
# These tests should be compatible with CPU-only environments.

# Ensure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:.

echo "Running CPU-only tests..."
pytest tests/cpu
