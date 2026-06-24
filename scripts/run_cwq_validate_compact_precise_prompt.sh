#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec "${PROJECT_ROOT}/scripts/run_cwq_validate_graphapi.sh" \
  --compact-precise-path-prompt \
  "$@"
