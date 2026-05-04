#!/usr/bin/env bash
# Launcher used by the agent to run a single calibration candidate on GPU 2.
# Args: $1 = step_idx (0..7)
set -euo pipefail

i="${1:?step index required}"
PYTHON_BIN="${PYTHON:-.venv/bin/python}"
TIMEOUT="${RUN_TIMEOUT_SECONDS:-900}"

case "$i" in
  0) desc="builtin FedAvg audit"; agg=(--aggregator default); name="algo_builtin_fedavg" ;;
  1) desc="explicit FedAvg audit"; agg=(--aggregator fedavg); name="algo_fedavg" ;;
  2) desc="FedProx mu 1e-5 [src: Li20 FedProx arXiv:1812.06127]"; agg=(--aggregator weighted --fedproxloss_mu 1e-5); name="algo_fedprox_1e5" ;;
  3) desc="FedProx mu 1e-4 [src: Li20 FedProx arXiv:1812.06127]"; agg=(--aggregator weighted --fedproxloss_mu 1e-4); name="algo_fedprox_1e4" ;;
  4) desc="FedAvgM lr1.0 m0.6 [src: Hsu19 FedAvgM arXiv:1909.06335]"; agg=(--aggregator fedavgm --server_lr 1.0 --server_momentum 0.6); name="algo_fedavgm_lr10_m06" ;;
  5) desc="FedAvgM lr2.0 m0.4 [src: Hsu19 FedAvgM arXiv:1909.06335]"; agg=(--aggregator fedavgm --server_lr 2.0 --server_momentum 0.4); name="algo_fedavgm_lr20_m04" ;;
  6) desc="FedAdam [src: Reddi20 FedOpt arXiv:2003.00295]"; agg=(--aggregator fedadam --server_lr 1.0 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 1e-3); name="algo_fedadam" ;;
  7) desc="SCAFFOLD metadata mode [src: Karimireddy20 SCAFFOLD arXiv:1910.06378]"; agg=(--aggregator scaffold); name="algo_scaffold" ;;
  *) echo "unknown step $i" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES=2 PYTHON="${PYTHON_BIN}" RUN_LOG="run_logs/${name}.log" RUN_TIMEOUT_SECONDS="${TIMEOUT}" \
  bash scripts/run_iteration.sh --description "${desc}" --target client.py -- \
  --n_clients 8 --num_rounds 10 --aggregation_epochs 4 --batch_size 64 --eval_batch_size 1024 \
  --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 \
  --final_eval_clients site-1 "${agg[@]}" --name "${name}"
