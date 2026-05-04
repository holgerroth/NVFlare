#!/usr/bin/env bash
# Generic launcher: launch_candidate.sh <name> <description> <extra args...>
set -euo pipefail

NAME="${1:?candidate name required}"
DESC="${2:?description required}"
shift 2

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
TIMEOUT="${RUN_TIMEOUT_SECONDS:-900}"
GPU="${CUDA_VISIBLE_DEVICES:-2}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHON="${PYTHON_BIN}" \
  PYTHONPYCACHEPREFIX="/tmp/auto-fl-pycache-${NAME}" \
  RUN_LOG="run_logs/${NAME}.log" RUN_TIMEOUT_SECONDS="${TIMEOUT}" \
  bash scripts/run_iteration.sh --description "${DESC}" --target client.py -- \
  --n_clients 8 --num_rounds 10 --aggregation_epochs 4 --batch_size 64 --eval_batch_size 1024 \
  --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 \
  --final_eval_clients site-1 "$@" --name "${NAME}"
