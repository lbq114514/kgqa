#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_BASE="${CONDA_BASE:-/home/ubuntu/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kgqa}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
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
SPLIT="${SPLIT:-test}"
INDEX="${INDEX:-0}"
SAMPLE_ID="${SAMPLE_ID:-}"
LOG_FILE="${LOG_FILE:-}"
VERBOSE="${VERBOSE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
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
    --sample-id)
      SAMPLE_ID="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CONFIG="$(resolve_project_path "$CONFIG")"

if [[ -n "$SAMPLE_ID" ]]; then
  INDEX="$(python - "$CONFIG" "$SPLIT" "$SAMPLE_ID" <<'PY'
import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
split = sys.argv[2]
sample_id = sys.argv[3]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
dataset_cfg = config.get("datasets", {}).get("cwq", {})
if split == "train":
    data_path = dataset_cfg.get("train_path")
elif split == "dev":
    data_path = dataset_cfg.get("dev_path") or dataset_cfg.get("validation_path")
else:
    data_path = dataset_cfg.get("test_path")
if not data_path:
    raise SystemExit(f"Could not resolve dataset path for split={split}")

path = Path(data_path)
for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    row_id = row.get("ID") or row.get("id") or row.get("sample_id")
    if row_id == sample_id:
        print(i)
        raise SystemExit(0)
raise SystemExit(f"Sample ID not found: {sample_id}")
PY
)"
fi

cmd=(
  python main.py
  --dataset cwq
  --mode single
  --split "$SPLIT"
  --index "$INDEX"
  --config "$CONFIG"
  --graphapi-enabled
)

if [[ "$VERBOSE" == "1" ]]; then
  echo "Working dir: $PROJECT_ROOT"
  echo "Python: $(command -v python)"
  echo "Config path: $CONFIG"
  echo "Command: ${cmd[*]}"
  echo "Environment:"
  echo "  CONDA_ENV_NAME=$CONDA_ENV_NAME"
  echo "  SPLIT=$SPLIT"
  echo "  INDEX=$INDEX"
  if [[ -n "$SAMPLE_ID" ]]; then
    echo "  SAMPLE_ID=$SAMPLE_ID"
  fi
  echo "Config summary:"
  python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = {
    "llm_model": config.get("llm", {}).get("model"),
    "graphapi_db_path": config.get("graphapi", {}).get("db_path"),
    "retrieval_index_dir": config.get("retrieval", {}).get("index_dir"),
    "entity_candidate_top_k": config.get("retrieval", {}).get("entity_candidate_top_k"),
    "entity_candidate_selected_top_k": config.get("retrieval", {}).get("entity_candidate_selected_top_k"),
    "subquestion_anchor_filtering": config.get("retrieval", {}).get("subquestion_anchor_filtering"),
    "qa_concurrency": config.get("evaluation", {}).get("qa_concurrency"),
}
for key, value in summary.items():
    print(f"  {key}={value}")
PY
fi

if [[ -n "$LOG_FILE" ]]; then
  LOG_FILE="$(resolve_project_path "$LOG_FILE")"
  mkdir -p "$(dirname "$LOG_FILE")"
  {
    echo "Working dir: $PROJECT_ROOT"
    echo "Python: $(command -v python)"
    echo "Config path: $CONFIG"
    echo "Command: ${cmd[*]}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %z')"
  } >"$LOG_FILE"
  nohup "${cmd[@]}" >>"$LOG_FILE" 2>&1 &
  pid=$!
  echo "Started cwq_single_relation_beam_direct"
  echo "PID: $pid"
  echo "Log: $LOG_FILE"
else
  "${cmd[@]}"
fi

echo "Config: $CONFIG"
echo "Index: $INDEX"
if [[ -n "$SAMPLE_ID" ]]; then
  echo "Sample ID: $SAMPLE_ID"
fi
echo "Split: $SPLIT"
echo "Conda env: $CONDA_ENV_NAME"
