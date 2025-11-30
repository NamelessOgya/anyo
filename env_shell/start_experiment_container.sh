#!/usr/bin/env bash
set -e

# Usage: ./start_experiment_container.sh [OPTIONS] [GPU_ID] [SUFFIX]
# 
# Arguments:
#   GPU_ID: GPU ID to use (default: 0)
#   SUFFIX: Suffix for container name (default: empty)
#
# Options:
#   --reset   : Force remove existing container and create a new one.
#   --install : Run 'poetry install' inside the container after starting.
#
# Examples:
#   ./start.sh                      # GPU 0, Default name
#   ./start.sh 1 "-gpu1"            # GPU 1, Name with suffix
#   ./start.sh --reset              # Recreate default container
#   ./start.sh --install 0          # GPU 0, run poetry install
#   ./start.sh --reset --install 1 "-gpu1" # Recreate GPU 1 container and install deps

RESET=false
INSTALL=false
GPU_ID=0
SUFFIX=""

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --reset)
      RESET=true
      shift
      ;;
    --install)
      INSTALL=true
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Restore positional args
set -- "${POSITIONAL_ARGS[@]}"
if [ -n "$1" ]; then GPU_ID=$1; fi
if [ -n "$2" ]; then SUFFIX=$2; fi

CONTAINER_NAME="ilora-dllm2rec-dev${SUFFIX}"
HOST_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
IMAGE_NAME="ilora-dllm2rec:latest"

echo "Target Container: ${CONTAINER_NAME} (GPU: ${GPU_ID})"

# 1. Build Image if missing
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
  echo "Image not found. Building..."
  docker build -t "${IMAGE_NAME}" "${HOST_PROJECT_ROOT}"
fi

# 2. Check Container Status
CONTAINER_STATUS=$(docker inspect --format="{{.State.Status}}" "${CONTAINER_NAME}" 2>/dev/null || echo "not_found")

# 3. Reset logic
if [ "$RESET" = true ] && [ "$CONTAINER_STATUS" != "not_found" ]; then
    echo "Reset requested. Removing existing container ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    CONTAINER_STATUS="not_found"
fi

# 4. Start or Run Container
if [ "${CONTAINER_STATUS}" = "running" ]; then
    echo "Container ${CONTAINER_NAME} is already running."
elif [ "${CONTAINER_STATUS}" = "exited" ] || [ "${CONTAINER_STATUS}" = "created" ]; then
    echo "Starting existing container ${CONTAINER_NAME}..."
    docker start "${CONTAINER_NAME}"
else
    echo "Creating and starting new container ${CONTAINER_NAME} on GPU ${GPU_ID}..."
    docker run -d \
      --gpus "device=${GPU_ID}" \
      --memory=200g \
      --name "${CONTAINER_NAME}" \
      -v "${HOST_PROJECT_ROOT}:/workspace" \
      -w /workspace \
      "${IMAGE_NAME}" \
      tail -f /dev/null
      
    # New container created, might want to install by default? 
    # For now, only if --install is passed or explicit logic.
    # But let's stick to flags.
fi

# 5. Install Dependencies if requested
if [ "$INSTALL" = true ]; then
    echo "Installing dependencies (poetry install)..."
    # Wait a bit if just started? Usually instant for 'tail -f /dev/null'
    docker exec "${CONTAINER_NAME}" bash -c "poetry lock && poetry install"
fi

# 6. Connect
echo "Connecting to container ${CONTAINER_NAME}..."
echo "To detach from tmux: Ctrl+b, d"
echo "To exit container: exit"
docker exec -it "${CONTAINER_NAME}" bash
