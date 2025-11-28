#!/usr/bin/env bash
set -e

CONTAINER_NAME="ilora-dllm2rec-dev"
HOST_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
IMAGE_NAME="ilora-dllm2rec:latest"

# イメージがなければビルド
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
  echo "Image not found. Building..."
  docker build -t "${IMAGE_NAME}" "${HOST_PROJECT_ROOT}"
fi

# 既存の同名コンテナがあれば停止・削除
if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Stopping and removing existing container..."
    docker stop "${CONTAINER_NAME}" > /dev/null
fi

echo "Starting container..."
docker run -it --rm \
  --gpus '"device=0"' \
  --memory=200g \
  --name "${CONTAINER_NAME}" \
  -v "${HOST_PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash
