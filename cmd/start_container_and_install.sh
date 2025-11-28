#!/usr/bin/env bash
set -e

CONTAINER_NAME="ilora-dev-container"
IMAGE_NAME="ilora-dllm2rec:latest"
HOST_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

echo "Starting container ${CONTAINER_NAME}..."

# コンテナが既に存在する場合は削除
if docker ps -a --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} already exists. Stopping and removing..."
  docker stop "${CONTAINER_NAME}" > /dev/null
  docker rm "${CONTAINER_NAME}" > /dev/null
fi

# イメージがなければビルド (Dockerfileはプロジェクトルートにあると仮定)
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
  echo "Docker image ${IMAGE_NAME} not found. Building..."
  docker build -t "${IMAGE_NAME}" "${HOST_PROJECT_ROOT}"
fi

# コンテナをバックグラウンドで起動
docker run -d \
  --name "${CONTAINER_NAME}" \
  -v "${HOST_PROJECT_ROOT}:/workspace" \
  -w /workspace \
  --gpus all \
  -it "${IMAGE_NAME}" \
  bash

echo "Container ${CONTAINER_NAME} started."
echo "Installing poetry dependencies inside the container (this may take a while for the first time)..."

# 依存関係のインストール (初回のみ)
# コンテナが完全に起動するまで少し待つ
sleep 5 
docker exec "${CONTAINER_NAME}" bash -c "poetry lock && poetry install"

echo "Poetry dependencies installed in ${CONTAINER_NAME}."
echo "To connect to the container, run: docker exec -it ${CONTAINER_NAME} bash"
