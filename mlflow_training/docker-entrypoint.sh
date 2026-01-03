#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "pytest" ]]; then
  if [[ ! -f artifacts/pipeline.pkl || ! -f artifacts/metrics.json ]]; then
    python train_mlflow.py
  fi
fi

exec "$@"
