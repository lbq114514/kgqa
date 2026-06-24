#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_BASE="${CONDA_BASE:-/home/ubuntu/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kgqa}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Some conda activation hooks assume CUDA_HOME may be unset and break under `set -u`.
set +u
conda activate "${CONDA_ENV_NAME}"
set -u

resolve_project_path() {
  local path_value="$1"
  if [[ "$path_value" == /* ]]; then
    printf '%s\n' "$path_value"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path_value}"
  fi
}

CONFIG="${CONFIG:-cwq_validate_relation_beam_guarded.yaml}"
QA_CONCURRENCY="${QA_CONCURRENCY:-}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOG_FILE="${LOG_FILE:-outputs/logs/cwq_validate_relation_beam_direct.log}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --qa-concurrency)
      QA_CONCURRENCY="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CONFIG="$(resolve_project_path "$CONFIG")"
LOG_FILE="$(resolve_project_path "$LOG_FILE")"
mkdir -p "$(dirname "$LOG_FILE")"

cmd=(
  python main.py
  --dataset cwq
  --mode validate
  --split "$SPLIT"
  --config "$CONFIG"
  --graphapi-enabled
)

if [[ -n "$QA_CONCURRENCY" ]]; then
  cmd+=(--qa-concurrency "$QA_CONCURRENCY")
fi

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi

nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
pid=$!

echo "Started cwq_validate_relation_beam_direct"
echo "PID: $pid"
echo "Log: $LOG_FILE"
echo "Config: $CONFIG"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  echo "Output dir override requested but not applied by script: $OUTPUT_DIR"
fi
echo "Conda env: $CONDA_ENV_NAME"
