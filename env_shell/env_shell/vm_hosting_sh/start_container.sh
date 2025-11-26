#!/bin/bash
set -e

# もし同名コンテナが残っていれば消す
docker rm -f t2r 2>/dev/null || true

# バックグラウンドで立ち上げ。tail で常駐させる
docker run -d \
  --gpus all \
  --name t2r \
  -v "$(pwd):/app" \
  -w "/app" \
  --restart unless-stopped \
  t4r \
  tail -f /dev/null

exec docker exec -it t2r bash