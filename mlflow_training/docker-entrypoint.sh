#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "pytest" ]]; then
  # Only run training before tests when explicitly requested.
  if [[ "${RUN_TRAIN_BEFORE_TESTS:-0}" == "1" ]]; then
    if [[ ! -f artifacts/pipeline.pkl || ! -f artifacts/metrics.json ]]; then
      python train_mlflow.py
    fi
  fi
fi

exec "$@"
