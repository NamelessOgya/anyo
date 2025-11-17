#!/usr/bin/env bash
PYTHONPATH=/workspace /opt/conda/bin/poetry run python -m src.exp.run_teacher "$@"
