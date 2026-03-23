#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/diffusor/bin/python}"
LOG_DIR="${LOG_DIR:-results/ppo_parallel_logs}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --grid-yaml config/IEEE33.yaml
  --time-yaml config/time.yaml
  --scenario-batch-root results/scenario_batch_1000ev_1w_mpc
  --train-scenarios S1,S2,S3,S4,S5,S6,S7,S8,S9
  --eval-scenario S0
  --scenario-lookahead-hours 24
  --total-timesteps 4000000
  --horizon-hours 168
  --resolution-minutes 5
  --ev-resolution-seconds 300
  --num-envs 1
  --num-eval-envs 1
  --vec-env dummy
  --device cpu
  --torch-num-threads 4
  --checkpoint-freq 100000
  --eval-freq 50000
  --eval-episodes 1
  --progress-bar
)

run_one() {
  local core_set="$1"
  local script_path="$2"
  local run_name="$3"
  local log_path="$LOG_DIR/${run_name}.log"
  local mpl_dir="/tmp/mplconfig_${run_name}"

  if command -v taskset >/dev/null 2>&1; then
    nohup env LC_ALL=C MPLCONFIGDIR="$mpl_dir" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
      taskset -c "$core_set" "$PYTHON_BIN" "$script_path" "${COMMON_ARGS[@]}" --run-name "$run_name" \
      >"$log_path" 2>&1 &
  else
    nohup env LC_ALL=C MPLCONFIGDIR="$mpl_dir" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
      "$PYTHON_BIN" "$script_path" "${COMMON_ARGS[@]}" --run-name "$run_name" \
      >"$log_path" 2>&1 &
  fi
  echo "$run_name -> $log_path"
}

run_one "0-3" "DRL/train_ppo.py"  "ppo_S1toS9_train_S0eval_24h_4M"
run_one "4-7" "DRL/train_sac.py"  "sac_S1toS9_train_S0eval_24h_4M"
run_one "8-11" "DRL/train_td3.py" "td3_S1toS9_train_S0eval_24h_4M"
run_one "12-15" "DRL/train_ddpg.py" "ddpg_S1toS9_train_S0eval_24h_4M"

echo "launched 4 algorithms in parallel"
