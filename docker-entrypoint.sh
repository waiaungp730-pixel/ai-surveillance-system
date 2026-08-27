#!/usr/bin/env bash
set -euo pipefail

# Require NVIDIA Container Toolkit on the host (docker run --gpus all).
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  echo "WARNING: nvidia-smi not found — inference will fall back to CPU."
fi

DATA_DIR="${DATA_DIR:-/app/data}"
mkdir -p "$DATA_DIR/captures" "$DATA_DIR/uploads" "$DATA_DIR/live_preview"

link_dir() {
  local name="$1"
  local target="/app/$name"
  local store="$DATA_DIR/$name"
  mkdir -p "$store"
  if [ ! -L "$target" ]; then
    rm -rf "$target"
    ln -sfn "$store" "$target"
  fi
}

link_dir captures
link_dir uploads

for db in visitor_tracking.db face_gallery.db; do
  store="$DATA_DIR/$db"
  target="/app/$db"
  if [ ! -e "$store" ]; then
    touch "$store" 2>/dev/null || true
  fi
  if [ ! -L "$target" ]; then
    rm -f "$target"
    ln -sfn "$store" "$target"
  fi
done

if [ -f "$DATA_DIR/cameras.json" ] && [ ! -f /app/cameras.json ]; then
  ln -sfn "$DATA_DIR/cameras.json" /app/cameras.json
fi

exec python3.12 run_system.py --device "${DEVICE:-cuda}" --skip-validation "$@"
