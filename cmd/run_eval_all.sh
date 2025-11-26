#!/usr/bin/env bash
PYTHONPATH=/workspace poetry run python -m src.exp.run_eval_all "$@"
