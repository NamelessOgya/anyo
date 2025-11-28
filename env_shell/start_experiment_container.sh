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

# コンテナの状態を確認
CONTAINER_STATUS=$(docker inspect --format="{{.State.Status}}" "${CONTAINER_NAME}" 2>/dev/null || echo "not_found")

if [ "${CONTAINER_STATUS}" = "running" ]; then
    echo "Container ${CONTAINER_NAME} is already running."
elif [ "${CONTAINER_STATUS}" = "exited" ] || [ "${CONTAINER_STATUS}" = "created" ]; then
    echo "Starting existing container ${CONTAINER_NAME}..."
    docker start "${CONTAINER_NAME}"
else
    echo "Creating and starting new container ${CONTAINER_NAME}..."
    docker run -d \
      --gpus '"device=0"' \
      --memory=200g \
      --name "${CONTAINER_NAME}" \
      -v "${HOST_PROJECT_ROOT}:/workspace" \
      -w /workspace \
      "${IMAGE_NAME}" \
      tail -f /dev/null
fi

echo "Connecting to container..."
echo "To detach from tmux: Ctrl+b, d"
echo "To exit container: exit"
docker exec -it "${CONTAINER_NAME}" bash
