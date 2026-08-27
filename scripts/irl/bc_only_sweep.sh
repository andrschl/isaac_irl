#!/usr/bin/env bash
# Comprehensive BC-only (weight=1) verification sweep.
#
# Runs pure behavioral cloning with weight pinned to 1.0. The runner uses
# `bc_only_update`, so no PPO rollouts or IRL reward updates are performed.
# The sweep varies BC loss type, actor/BC learning rate, and seed.
#
# Runs land under:
#   logs/irl/bc_only_<loss>_lr<lr>/<timestamp>_<run_name>/
# and (for wandb) are grouped per experiment_name.
#
# Usage:
#   scripts/irl/bc_only_sweep.sh                          # full grid, defaults
#   SEEDS="1 2 3" scripts/irl/bc_only_sweep.sh            # more seeds
#   LOSS_TYPES="nll" LEARNING_RATES="1e-3" \
#     scripts/irl/bc_only_sweep.sh                        # narrow to one combo
#   LOGGER=noop scripts/irl/bc_only_sweep.sh       # no wandb
#
# Knobs (env vars):
#   LOSS_TYPES        space-separated BC loss types to sweep   (default: "nll mse")
#   LEARNING_RATES    space-separated actor/BC learning rates  (default: "1e-3 5e-3 1e-2 5e-2")
#   SEEDS             space-separated seeds                    (default: "1 2 3")
#   MAX_ITERATIONS    BC iterations per run                    (default: 20000)
#   VALIDATION_INTERVAL  env-rollout validation cadence (iters) (default: 1000).
#                     BC iters are ms each; validation triggers a full env rollout,
#                     so bump high so it doesn't dominate wall-clock. 0 disables.
#   TASK              Isaac Lab task name                      (default: Isaac-Lift-Cube-Franka-v0)
#   EXPERT_DATA_PATH  HDF5 with (obs, actions, features)       (default: logs/demos/franka_lift/demos.hdf5)
#   LOGGER            wandb | noop            (default: wandb)
#   WANDB_PROJECT     logger project name                      (default: bc_only_verification)
#   EXTRA_ARGS        base args appended to every run           (default: --headless)
#   PYTHON            python interpreter                       (default: python)
#
# Total runs = |LOSS_TYPES| × |LEARNING_RATES| × |SEEDS|, run sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LOSS_TYPES="${LOSS_TYPES:-nll mse}"
LEARNING_RATES="${LEARNING_RATES:-1e-3 5e-3 1e-2 5e-2}"
SEEDS="${SEEDS:-1 2 3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-1000}"
TASK="${TASK:-Isaac-Lift-Cube-Franka-v0}"
EXPERT_DATA_PATH="${EXPERT_DATA_PATH:-logs/demos/franka_lift/demos.hdf5}"
LOGGER="${LOGGER:-wandb}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-${WANDB_PROJECT:-bc_only_verification}}"
BASE_EXTRA_ARGS="${EXTRA_ARGS:---headless}"
PYTHON="${PYTHON:-python}"

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_float() {
  awk -v value="$1" 'BEGIN {
    if (value !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) exit 1
    exit !((value + 0) > 0)
  }'
}

if ! is_positive_int "${MAX_ITERATIONS}"; then
  echo "[ERROR] MAX_ITERATIONS must be a positive integer (got '${MAX_ITERATIONS}')." >&2
  exit 1
fi
if ! is_non_negative_int "${VALIDATION_INTERVAL}"; then
  echo "[ERROR] VALIDATION_INTERVAL must be a non-negative integer (got '${VALIDATION_INTERVAL}')." >&2
  exit 1
fi

case "${LOGGER}" in
  wandb|noop) ;;
  *) echo "[ERROR] LOGGER must be wandb or noop (got '${LOGGER}')." >&2; exit 1 ;;
esac

for LOSS in ${LOSS_TYPES}; do
  case "${LOSS}" in
    nll|mse) ;;
    *) echo "[ERROR] LOSS_TYPES entries must be 'nll' or 'mse' (got '${LOSS}')." >&2; exit 1 ;;
  esac
done

for LR in ${LEARNING_RATES}; do
  if ! is_positive_float "${LR}"; then
    echo "[ERROR] LEARNING_RATES entries must be positive numbers (got '${LR}')." >&2
    exit 1
  fi
done

for SEED in ${SEEDS}; do
  if ! is_non_negative_int "${SEED}"; then
    echo "[ERROR] SEEDS entries must be non-negative integers (got '${SEED}')." >&2
    exit 1
  fi
done

TOTAL_RUNS=0
for _ in ${LOSS_TYPES}; do
  for _ in ${LEARNING_RATES}; do
    for _ in ${SEEDS}; do TOTAL_RUNS=$((TOTAL_RUNS + 1)); done
  done
done

RUN_IDX=0
FAILED_RUNS=()

echo "================================================================="
echo " BC-only verification sweep"
echo "================================================================="
echo " task           : ${TASK}"
echo " expert data    : ${EXPERT_DATA_PATH}"
echo " loss types     : ${LOSS_TYPES}"
echo " learning rates : ${LEARNING_RATES}"
echo " seeds          : ${SEEDS}"
echo " max iterations : ${MAX_ITERATIONS}"
echo " validation int.: ${VALIDATION_INTERVAL}"
echo " logger         : ${LOGGER}"
if [[ "${LOGGER}" == "wandb" ]]; then
  echo " log project    : ${LOG_PROJECT_NAME}"
fi
echo " total runs     : ${TOTAL_RUNS}"
echo "================================================================="
echo ""

for LOSS in ${LOSS_TYPES}; do
  for LR in ${LEARNING_RATES}; do
    LR_TAG="${LR//./p}"
    EXPERIMENT_NAME="bc_only_${LOSS}_lr${LR_TAG}"

    for SEED in ${SEEDS}; do
      RUN_IDX=$((RUN_IDX + 1))
      RUN_NAME="${LOSS}_lr${LR_TAG}_weight_1.0_seed_${SEED}"

      echo "-----------------------------------------------------------------"
      echo " [${RUN_IDX}/${TOTAL_RUNS}] loss=${LOSS}  lr=${LR}  seed=${SEED}  run=${RUN_NAME}"
      echo "-----------------------------------------------------------------"

      CMD=(
        "${PYTHON}" scripts/irl/train_irl.py
        --task "${TASK}"
        --expert_data_path "${EXPERT_DATA_PATH}"
        --bc_weight "1.0"
        --bc_loss_type "${LOSS}"
        --seed "${SEED}"
        --max_iterations "${MAX_ITERATIONS}"
        --validation_interval "${VALIDATION_INTERVAL}"
        --experiment_name "${EXPERIMENT_NAME}"
        --run_name "${RUN_NAME}"
        --logger "${LOGGER}"
      )
      if [[ "${LOGGER}" == "wandb" ]]; then
        CMD+=(--log_project_name "${LOG_PROJECT_NAME}")
      fi
      # shellcheck disable=SC2086
      CMD+=(${BASE_EXTRA_ARGS} --learning_rate "${LR}")

      echo "+ ${CMD[*]}"

      RUN_FAILED=0
      if ! "${CMD[@]}"; then
        RUN_FAILED=1
      fi

      if [[ ${RUN_FAILED} -ne 0 ]]; then
        echo "[WARN] Run loss=${LOSS} lr=${LR} seed=${SEED} failed; continuing sweep." >&2
        FAILED_RUNS+=("loss=${LOSS} lr=${LR} seed=${SEED}")
      fi
      echo ""
    done
  done
done

echo "================================================================="
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
  echo " BC-only verification sweep complete: all ${TOTAL_RUNS} runs succeeded."
else
  echo " BC-only verification sweep complete: ${#FAILED_RUNS[@]}/${TOTAL_RUNS} runs FAILED:"
  for failed in "${FAILED_RUNS[@]}"; do
    echo "  - ${failed}"
  done
fi
echo " Loss types     : ${LOSS_TYPES}"
echo " Learning rates : ${LEARNING_RATES}"
echo " Seeds          : ${SEEDS}"
echo " Max iterations : ${MAX_ITERATIONS}"
echo " Logs           : logs/irl/bc_only_<loss>_lr<lr>/"
echo " View           : wandb dashboard for project ${LOG_PROJECT_NAME} (when LOGGER=wandb)"
echo "================================================================="

if [[ ${#FAILED_RUNS[@]} -ne 0 ]]; then
  exit 1
fi
