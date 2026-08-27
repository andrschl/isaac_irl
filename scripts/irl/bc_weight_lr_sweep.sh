#!/usr/bin/env bash
# Sweep BC mixing weight `bc_weight` × local learning-rate neighborhood × seed,
# with the BC loss type pinned to NLL.
#
# The learning rate at each bc_weight is the geometric interpolation between two
# pre-tuned endpoint LRs, optionally scaled by a local multiplier m:
#
#   eta_init(bc_weight) = eta_ppo^(1 - bc_weight) * eta_bc^(bc_weight)
#   eta_run(bc_weight, m) = m * eta_init(bc_weight)
#
# where eta_ppo (bc_weight=0 endpoint) is the vanilla PPO LR from
# configs/franka_lift/experiment.yaml and eta_bc (bc_weight=1 endpoint) is the best
# BC-only LR found via scripts/irl/bc_only_sweep.sh. Multipliers default to
# {0.5, 1, 2} so each bc_weight gets a 3-point local LR neighborhood.
#
# Self-contained: invokes scripts/irl/train_irl.py directly. BC shares PPO's
# actor optimizer, so `--learning_rate <lr>` controls both losses. Runs are
# sequential (Isaac Lab opens one SimulationApp per process — do NOT parallelize).
#
# Runs land under:
#   logs/irl/<EXPERIMENT_NAME>/<timestamp>_nll_mult<m>_bc_weight_<bc_weight>_seed_<s>/
# and (for wandb) are grouped under one project / one experiment_name.
#
# Usage:
#   scripts/irl/bc_weight_lr_sweep.sh                                 # full grid
#   BC_WEIGHTS="0 0.5 1" scripts/irl/bc_weight_lr_sweep.sh                # narrow bc_weight grid
#   LR_MULTIPLIERS="1" SEEDS="42" scripts/irl/bc_weight_lr_sweep.sh   # single LR, single seed
#   ETA_BC=1e-2 scripts/irl/bc_weight_lr_sweep.sh                     # different bc_weight=1 endpoint
#   LOGGER=noop scripts/irl/bc_weight_lr_sweep.sh              # no wandb
#
# Knobs (env vars):
#   BC_WEIGHTS            space-separated BC bc_weight values    (default: "0 0.01 0.1 0.9 0.99 1")
#   LR_MULTIPLIERS    space-separated local LR scales    (default: "0.5 1 2")
#   SEEDS             space-separated seeds              (default: "1 2 3")
#   ETA_PPO           bc_weight=0 endpoint learning rate     (default: 1e-4)
#   ETA_BC            bc_weight=1 endpoint learning rate     (default: 1e-3)
#   MAX_ITERATIONS    iterations per non-BC-only run      (default: 500)
#   BC_ONLY_MAX_ITERATIONS
#                     iterations for bc_weight=1 pure BC runs (default: 20000)
#   BC_ONLY_VALIDATION_INTERVAL
#                     validation cadence for bc_weight=1 runs (default: 1000).
#                     BC iters are ms each, but each validation triggers an env
#                     rollout — bump high so validation doesn't dominate wall-clock.
#   LOGGER            wandb | noop      (default: wandb)
#   WANDB_PROJECT     wandb project name (alias: LOG_PROJECT_NAME) (default: bc_weight_lr_sweep)
#   EXPERIMENT_NAME   group name for logs + logger       (default: bc_weight_lr_sweep_nll)
#   TASK              Isaac Lab task name                (default: Isaac-Lift-Cube-Franka-v0)
#   EXPERT_DATA_PATH  HDF5 with (obs, actions, features) (default: logs/demos/franka_lift/demos.hdf5)
#   RECORD_VIDEO      1 to record + save final clips     (default: 1)
#   VIDEO_LENGTH      env steps in the final clip        (default: 250 ≈ 5 s at 50 Hz)
#   EXTRA_ARGS        base args appended to every run    (default: --headless)
#   PYTHON            python interpreter                 (default: python; "echo" for dry-run)
#
# Total runs = |BC_WEIGHTS| × |LR_MULTIPLIERS| × |SEEDS|, run sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------
BC_WEIGHTS="${BC_WEIGHTS:-0 0.01 0.1 0.9 0.99 1}"
LR_MULTIPLIERS="${LR_MULTIPLIERS:-0.5 1 2}"
SEEDS="${SEEDS:-1 2 3}"
ETA_PPO="${ETA_PPO:-1e-4}"
ETA_BC="${ETA_BC:-5e-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
BC_ONLY_MAX_ITERATIONS="${BC_ONLY_MAX_ITERATIONS:-20000}"
BC_ONLY_VALIDATION_INTERVAL="${BC_ONLY_VALIDATION_INTERVAL:-1000}"
LOGGER="${LOGGER:-wandb}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-${WANDB_PROJECT:-bc_weight_lr_sweep}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bc_weight_lr_sweep_nll}"
TASK="${TASK:-Isaac-Lift-Cube-Franka-v0}"
EXPERT_DATA_PATH="${EXPERT_DATA_PATH:-logs/demos/franka_lift/demos.hdf5}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-250}"
BASE_EXTRA_ARGS="${EXTRA_ARGS:---headless}"
PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------
is_positive_float() {
  awk -v value="$1" 'BEGIN {
    if (value !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) exit 1
    exit !((value + 0) > 0)
  }'
}

is_valid_bc_weight() {
  awk -v bc_weight="$1" 'BEGIN {
    if (bc_weight !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) exit 1
    value = bc_weight + 0
    exit !(0 <= value && value <= 1)
  }'
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# ---------------------------------------------------------------------------
# Sanity checks.
# ---------------------------------------------------------------------------
if [[ ! -f "${EXPERT_DATA_PATH}" ]]; then
  echo "[ERROR] EXPERT_DATA_PATH does not exist: ${EXPERT_DATA_PATH}" >&2
  echo "        Record demos first via scripts/recording/record_synthetic_demos.py," >&2
  echo "        or set EXPERT_DATA_PATH=<path-to-demos.hdf5>." >&2
  exit 1
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "[ERROR] Python interpreter '${PYTHON}' not found on PATH." >&2
  exit 1
fi

case "${LOGGER}" in
  wandb|noop) ;;
  *)
    echo "[ERROR] LOGGER must be one of: wandb, noop (got '${LOGGER}')." >&2
    exit 1
    ;;
esac

case "${RECORD_VIDEO}" in
  0|1) ;;
  *)
    echo "[ERROR] RECORD_VIDEO must be 0 or 1 (got '${RECORD_VIDEO}')." >&2
    exit 1
    ;;
esac

if ! is_positive_int "${VIDEO_LENGTH}"; then
  echo "[ERROR] VIDEO_LENGTH must be a positive integer env-step count (got '${VIDEO_LENGTH}')." >&2
  exit 1
fi

if [[ -n "${MAX_ITERATIONS}" ]] && ! is_positive_int "${MAX_ITERATIONS}"; then
  echo "[ERROR] MAX_ITERATIONS must be a positive integer when set (got '${MAX_ITERATIONS}')." >&2
  exit 1
fi
if [[ -n "${BC_ONLY_MAX_ITERATIONS}" ]] && ! is_positive_int "${BC_ONLY_MAX_ITERATIONS}"; then
  echo "[ERROR] BC_ONLY_MAX_ITERATIONS must be a positive integer when set (got '${BC_ONLY_MAX_ITERATIONS}')." >&2
  exit 1
fi
if [[ -n "${BC_ONLY_VALIDATION_INTERVAL}" ]] && ! is_non_negative_int "${BC_ONLY_VALIDATION_INTERVAL}"; then
  echo "[ERROR] BC_ONLY_VALIDATION_INTERVAL must be a non-negative integer when set (got '${BC_ONLY_VALIDATION_INTERVAL}')." >&2
  exit 1
fi

for ENDPOINT in ETA_PPO ETA_BC; do
  if ! is_positive_float "${!ENDPOINT}"; then
    echo "[ERROR] ${ENDPOINT} must be a positive numeric value (got '${!ENDPOINT}')." >&2
    exit 1
  fi
done

for BC_WEIGHT in ${BC_WEIGHTS}; do
  if ! is_valid_bc_weight "${BC_WEIGHT}"; then
    echo "[ERROR] BC_WEIGHTS entries must be numeric values in [0, 1] (got '${BC_WEIGHT}')." >&2
    exit 1
  fi
done

for MULT in ${LR_MULTIPLIERS}; do
  if ! is_positive_float "${MULT}"; then
    echo "[ERROR] LR_MULTIPLIERS entries must be positive numbers (got '${MULT}')." >&2
    exit 1
  fi
done

for SEED in ${SEEDS}; do
  if ! is_non_negative_int "${SEED}"; then
    echo "[ERROR] SEEDS entries must be non-negative integers (got '${SEED}')." >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Summary (compute the (bc_weight, m) -> lr grid up front so the user sees it).
# ---------------------------------------------------------------------------
TOTAL_RUNS=0
for _ in ${BC_WEIGHTS}; do
  for _ in ${LR_MULTIPLIERS}; do
    for _ in ${SEEDS}; do TOTAL_RUNS=$((TOTAL_RUNS + 1)); done
  done
done

echo "================================================================="
echo " BC bc_weight × LR-neighborhood × seed sweep  (loss=nll)"
echo "================================================================="
echo " task            : ${TASK}"
echo " expert data     : ${EXPERT_DATA_PATH}"
echo " bc_weights          : ${BC_WEIGHTS}"
echo " lr multipliers  : ${LR_MULTIPLIERS}"
echo " seeds           : ${SEEDS}"
echo " eta_ppo (a=0)   : ${ETA_PPO}"
echo " eta_bc  (a=1)   : ${ETA_BC}"
echo " logger          : ${LOGGER}"
if [[ "${LOGGER}" == "wandb" ]]; then
  echo " log project     : ${LOG_PROJECT_NAME}"
fi
echo " experiment name : ${EXPERIMENT_NAME}"
if [[ -n "${MAX_ITERATIONS}" ]]; then
  echo " max iterations  : ${MAX_ITERATIONS}"
fi
echo " bc-only iters   : ${BC_ONLY_MAX_ITERATIONS}"
echo " bc-only val.int.: ${BC_ONLY_VALIDATION_INTERVAL}"
if [[ "${RECORD_VIDEO}" == "1" ]]; then
  echo " final video     : ${VIDEO_LENGTH} env steps (~$(awk "BEGIN{printf \"%.1f\", ${VIDEO_LENGTH}/50}") s at 50 Hz)"
else
  echo " final video     : disabled (RECORD_VIDEO=0)"
fi
echo " extra args base : ${BASE_EXTRA_ARGS}"
echo " total runs      : ${TOTAL_RUNS}"
echo "-----------------------------------------------------------------"
printf " %-8s %-14s" "bc_weight" "eta_init"
for MULT in ${LR_MULTIPLIERS}; do
  printf " %-14s" "m=${MULT}"
done
printf "\n"
for BC_WEIGHT in ${BC_WEIGHTS}; do
  ETA_INIT=$(awk -v a="${BC_WEIGHT}" -v ep="${ETA_PPO}" -v eb="${ETA_BC}" \
    'BEGIN { printf "%.6e", (ep^(1-a)) * (eb^a) }')
  printf " %-8s %-14s" "${BC_WEIGHT}" "${ETA_INIT}"
  for MULT in ${LR_MULTIPLIERS}; do
    LR=$(awk -v eta="${ETA_INIT}" -v m="${MULT}" \
      'BEGIN { printf "%.6e", m * eta }')
    printf " %-14s" "${LR}"
  done
  printf "\n"
done
echo "================================================================="
echo ""

# ---------------------------------------------------------------------------
# Sweep loop.
# ---------------------------------------------------------------------------
NUM_SEEDS=$(awk -v s="${SEEDS}" 'BEGIN { n = split(s, _, /[[:space:]]+/); print n }')
RUN_IDX=0
FAILED_RUNS=()

for BC_WEIGHT in ${BC_WEIGHTS}; do
  ETA_INIT=$(awk -v a="${BC_WEIGHT}" -v ep="${ETA_PPO}" -v eb="${ETA_BC}" \
    'BEGIN { printf "%.6e", (ep^(1-a)) * (eb^a) }')

  for MULT in ${LR_MULTIPLIERS}; do
    LR=$(awk -v eta="${ETA_INIT}" -v m="${MULT}" \
      'BEGIN { printf "%.6e", m * eta }')
    MULT_TAG="${MULT//./p}"

    for SEED in ${SEEDS}; do
      RUN_IDX=$((RUN_IDX + 1))

      if [[ "${NUM_SEEDS}" -eq 1 ]]; then
        RUN_NAME="nll_mult${MULT_TAG}_bc_weight_${BC_WEIGHT}"
      else
        RUN_NAME="nll_mult${MULT_TAG}_bc_weight_${BC_WEIGHT}_seed_${SEED}"
      fi

      echo "-----------------------------------------------------------------"
      echo " [${RUN_IDX}/${TOTAL_RUNS}] bc_weight=${BC_WEIGHT}  m=${MULT}  lr=${LR}  seed=${SEED}  run=${RUN_NAME}"
      echo "-----------------------------------------------------------------"

      CMD=(
        "${PYTHON}" scripts/irl/train_irl.py
        --task "${TASK}"
        --expert_data_path "${EXPERT_DATA_PATH}"
        --bc_weight "${BC_WEIGHT}"
        --bc_loss_type "nll"
        --seed "${SEED}"
        --experiment_name "${EXPERIMENT_NAME}"
        --run_name "${RUN_NAME}"
        --logger "${LOGGER}"
      )
      if [[ "${LOGGER}" == "wandb" ]]; then
        CMD+=(--log_project_name "${LOG_PROJECT_NAME}")
      fi
      RUN_MAX_ITERATIONS="${MAX_ITERATIONS}"
      RUN_VALIDATION_INTERVAL=""
      if awk -v bc_weight="${BC_WEIGHT}" 'BEGIN { exit !((bc_weight + 0) == 1.0) }'; then
        RUN_MAX_ITERATIONS="${BC_ONLY_MAX_ITERATIONS}"
        # BC-only iters are ~ms each, but validation triggers a full env rollout —
        # bump the cadence so validation doesn't dominate wall-clock.
        RUN_VALIDATION_INTERVAL="${BC_ONLY_VALIDATION_INTERVAL}"
      fi
      if [[ -n "${RUN_MAX_ITERATIONS}" ]]; then
        CMD+=(--max_iterations "${RUN_MAX_ITERATIONS}")
      fi
      if [[ -n "${RUN_VALIDATION_INTERVAL}" ]]; then
        CMD+=(--validation_interval "${RUN_VALIDATION_INTERVAL}")
      fi
      if [[ "${RECORD_VIDEO}" == "1" ]]; then
        CMD+=(--video --video_final_only --video_length "${VIDEO_LENGTH}")
      fi
      # shellcheck disable=SC2086
      CMD+=(${BASE_EXTRA_ARGS} --learning_rate "${LR}")

      echo "+ ${CMD[*]}"

      # Snapshot the experiment log dir before the run so we can pick out the
      # new run subfolder regardless of the timestamped name train_irl.py picks.
      LOG_PARENT="logs/irl/${EXPERIMENT_NAME}"
      mkdir -p "${LOG_PARENT}"
      PRE_DIRS="$(ls -1 "${LOG_PARENT}" 2>/dev/null || true)"

      RUN_FAILED=0
      if ! "${CMD[@]}"; then
        RUN_FAILED=1
      fi

      # Isaac Sim's SimulationApp.close() can call os._exit(0) during shutdown
      # after a Python traceback, masking the failure with exit code 0. Detect
      # this by checking whether the run actually produced a model checkpoint.
      NEW_DIRS="$(ls -1 "${LOG_PARENT}" 2>/dev/null || true)"
      NEW_RUN_DIR=""
      while IFS= read -r d; do
        [[ -z "$d" ]] && continue
        if ! grep -Fxq -- "$d" <<<"${PRE_DIRS}"; then
          NEW_RUN_DIR="${LOG_PARENT}/$d"
          break
        fi
      done <<<"${NEW_DIRS}"

      if [[ ${RUN_FAILED} -eq 0 ]]; then
        if [[ -z "${NEW_RUN_DIR}" ]]; then
          echo "[WARN] No new run directory under ${LOG_PARENT}; treating as failure." >&2
          RUN_FAILED=1
        elif ! compgen -G "${NEW_RUN_DIR}/model_*.pt" >/dev/null; then
          echo "[WARN] Run dir ${NEW_RUN_DIR} has no model_*.pt checkpoint; treating as failure (likely silent crash via SimulationApp.close)." >&2
          RUN_FAILED=1
        fi
      fi

      if [[ ${RUN_FAILED} -ne 0 ]]; then
        echo "[WARN] Run bc_weight=${BC_WEIGHT} m=${MULT} seed=${SEED} failed; continuing sweep." >&2
        FAILED_RUNS+=("bc_weight=${BC_WEIGHT} m=${MULT} seed=${SEED}")
      fi
      echo ""
    done
  done
done

# ---------------------------------------------------------------------------
# Final report.
# ---------------------------------------------------------------------------
echo "================================================================="
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
  echo " Sweep complete: all ${TOTAL_RUNS} runs succeeded."
else
  echo " Sweep complete: ${#FAILED_RUNS[@]}/${TOTAL_RUNS} runs FAILED:"
  for failed in "${FAILED_RUNS[@]}"; do
    echo "   - ${failed}"
  done
fi
echo " Logs:  logs/irl/${EXPERIMENT_NAME}/"
echo " View:  wandb dashboard for project ${LOG_PROJECT_NAME} (when LOGGER=wandb)"
echo "================================================================="

[[ ${#FAILED_RUNS[@]} -eq 0 ]]
