#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_BASE="${CONDA_BASE:-/home/ubuntu/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kgqa}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

resolve_project_path() {
  local path_value="$1"
  if [[ "$path_value" == /* ]]; then
    printf '%s\n' "$path_value"
  else
    printf '%s\n' "${PROJECT_ROOT}/${path_value}"
  fi
}

BASE_CONFIG="${BASE_CONFIG:-config.yaml}"
SPLIT="${SPLIT:-test}"
INDEX="${INDEX:-0}"
COMPACT_PRECISE_PATH_PROMPT="${COMPACT_PRECISE_PATH_PROMPT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --index)
      INDEX="$2"
      shift 2
      ;;
    --compact-precise-path-prompt)
      COMPACT_PRECISE_PATH_PROMPT=1
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --runtime-config)
      RUNTIME_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

RUN_NAME="cwq_single_graphapi_index${INDEX}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-outputs/logs/${RUN_NAME}.log}"
RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-outputs/runtime_configs}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${RUNTIME_CONFIG_DIR}/${RUN_NAME}.yaml}"

OUTPUT_DIR="$(resolve_project_path "$OUTPUT_DIR")"
LOG_FILE="$(resolve_project_path "$LOG_FILE")"
RUNTIME_CONFIG="$(resolve_project_path "$RUNTIME_CONFIG")"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$RUNTIME_CONFIG")" "$OUTPUT_DIR"

python - "$BASE_CONFIG" "$RUNTIME_CONFIG" "$OUTPUT_DIR" "$COMPACT_PRECISE_PATH_PROMPT" <<'PY'
import sys
from pathlib import Path
import yaml

base_config = Path(sys.argv[1])
runtime_config = Path(sys.argv[2])
output_dir = sys.argv[3]
compact_precise_path_prompt = sys.argv[4] == "1"

config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
config.setdefault("graphapi", {})["enabled"] = True
config.setdefault("pruning", {})["compact_precise_path_prompt"] = compact_precise_path_prompt
config.setdefault("evaluation", {})["output_dir"] = output_dir
config["evaluation"]["sample_output_dir"] = "sample_predictions"
runtime_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

cmd=(
  python main.py
  --dataset cwq
  --mode single
  --split "$SPLIT"
  --index "$INDEX"
  --config "$RUNTIME_CONFIG"
  --graphapi-enabled
)

if [[ "$COMPACT_PRECISE_PATH_PROMPT" == "1" ]]; then
  cmd+=(--compact-precise-path-prompt)
fi

nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
pid=$!

echo "Started ${RUN_NAME}"
echo "PID: $pid"
echo "Log: $LOG_FILE"
echo "Runtime config: $RUNTIME_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "Conda env: $CONDA_ENV_NAME"
echo "Compact precise path prompt: $COMPACT_PRECISE_PATH_PROMPT"
