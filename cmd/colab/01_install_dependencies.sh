#!/bin/bash
# 01_install_dependencies.sh
# Installs project dependencies using poetry.

set -eu

echo "Lock file is out of sync. Running 'poetry lock'..."
poetry lock
echo "Installing/updating project dependencies with poetry..."
poetry install --no-root
echo "Dependency installation complete."