#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/bert-demo/runs/$(date -u +%Y%m%d-%H%M%S)}"
exec "$PROJECT_ROOT/.venv/bin/python" runpod/train_cuda.py "$@"
