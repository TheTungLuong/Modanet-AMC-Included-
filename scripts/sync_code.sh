#!/usr/bin/env bash
# Đồng bộ CODE local -> server, một chiều: code trên server giống hệt local (file thừa bị xoá).
# KHÔNG đụng tới data_processed/, outputs/, dist/, *.pt trên server.
#   bash scripts/sync_code.sh                  # host "vast" trong ~/.ssh/config, thư mục /workspace/modanet
#   bash scripts/sync_code.sh -n               # chạy thử: chỉ liệt kê file sẽ gửi / xoá
#   HOST=root@<IP> PORT=<PORT> bash scripts/sync_code.sh
#   DEST=/root/modanet bash scripts/sync_code.sh
# Trong VS Code: Ctrl+Shift+B (task "Sync code → server" trong .vscode/tasks.json).
# ⚠️  Code sửa trực tiếp trên server sẽ bị ghi đè — hãy sửa ở local rồi sync.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HOST:-vast}
DEST=${DEST:-/workspace/modanet}
SSH="ssh -o ConnectTimeout=10"
[ -n "${PORT:-}" ] && SSH="$SSH -p $PORT"
rsync -az --delete --itemize-changes --timeout=60 -e "$SSH" \
      --exclude=/data_processed --exclude=/outputs --exclude=/dist \
      --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.pt' --exclude='*.tmp' \
      "$@" ./ "$HOST:$DEST/"
echo "✓ Đã đồng bộ code -> $HOST:$DEST  ($(date +%H:%M:%S))"
