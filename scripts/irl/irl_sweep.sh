#!/usr/bin/env bash
# Random hyperparameter search over IRL-side knobs, with alpha pinned to 0
# (pure PPO+IRL, no BC).
#
# Sweep dimensions (drawn jointly via a single reproducible shuffle):
#   reward_learning_rate           (irl.reward_learning_rate)
#   reward_regularization_strength (reward.regularization_strength)
#   reward_updates_per_cycle       (runner.reward_updates_per_cycle)
#   rl_updates_per_cycle           (runner.rl_updates_per_cycle)
#
# NUM_SAMPLES tuples are drawn WITHOUT REPLACEMENT from the Cartesian grid of
# the four candidate lists, then each tuple is run at every seed in SEEDS.
# Total = NUM_SAMPLES × |SEEDS|, sequential (Isaac Lab opens one SimulationApp
# per process — do NOT parallelize).
#
# Each run lands under:
#   logs/irl/<EXPERIMENT_NAME>/<timestamp>_irl_s<idx>_seed_<s>/
# and a manifest is written once at the top of the sweep:
#   logs/irl/<EXPERIMENT_NAME>/sweep_manifest.csv
#
# Usage:
#   scripts/irl/irl_sweep.sh                                         # full sweep
#   NUM_SAMPLES=5 SEEDS="1" scripts/irl/irl_sweep.sh                 # smoke
#   SAMPLE_SEED=42 scripts/irl/irl_sweep.sh                          # different shuffle
#   REWARD_LRS="1e-4 1e-3" scripts/irl/irl_sweep.sh                  # custom grid
#   LOGGER=noop scripts/irl/irl_sweep.sh                      # no wandb
#
# Knobs (env vars):
#   REWARD_LRS         space-separated reward LR grid       (default: "1e-5 3e-5 1e-4 3e-4 1e-3")
#   REG_STRENGTHS      space-separated reg-strength grid    (default: "1e-6 1e-5 1e-4 1e-3 1e-2")
#   REWARD_UPDATES     space-separated reward updates/cycle (default: "1 2 4 8")
#   POLICY_UPDATES     space-separated policy updates/cycle (default: "1 2 4")
#   NUM_SAMPLES        # tuples drawn (clamped to grid size) (default: 20)
#   SAMPLE_SEED        RNG seed for the shuffle             (default: 0)
#   SEEDS              training seeds                       (default: "1 2 3")
#   MAX_ITERATIONS     iterations per run (unset => yaml)   (default: 500)
#   LOGGER             wandb | noop        (default: wandb)
#   WANDB_PROJECT      wandb project (alias: LOG_PROJECT_NAME) (default: irl_sweep)
#   EXPERIMENT_NAME    group name for logs + logger         (default: irl_sweep)
#   TASK               Isaac Lab task name                  (default: Isaac-Lift-Cube-Franka-v0)
#   EXPERT_DATA_PATH   HDF5 with (obs, actions, features)   (default: logs/demos/franka_lift/demos.hdf5)
#   RECORD_VIDEO       1 to record + save final clips       (default: 1)
#   VIDEO_LENGTH       env steps in the final clip          (default: 250 ≈ 5 s at 50 Hz)
#   EXTRA_ARGS         base args appended to every run      (default: --headless)
#   PYTHON             python interpreter                   (default: python; "echo" for dry-run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------
REWARD_LRS="${REWARD_LRS:-1e-5 3e-5 1e-4 3e-4 1e-3}"
REG_STRENGTHS="${REG_STRENGTHS:-1e-6 1e-5 1e-4 1e-3 1e-2}"
REWARD_UPDATES="${REWARD_UPDATES:-1 2 4 8}"
POLICY_UPDATES="${POLICY_UPDATES:-1 2 4}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
SEEDS="${SEEDS:-1 2 3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
LOGGER="${LOGGER:-wandb}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-${WANDB_PROJECT:-irl_sweep}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-irl_sweep}"
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

is_non_negative_float() {
  awk -v value="$1" 'BEGIN {
    if (value !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) exit 1
    exit !((value + 0) >= 0)
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
  exit 1
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "[ERROR] Python interpreter '${PYTHON}' not found on PATH." >&2
  exit 1
fi

case "${LOGGER}" in
  wandb|noop) ;;
  *) echo "[ERROR] LOGGER must be one of: wandb, noop (got '${LOGGER}')." >&2; exit 1 ;;
esac

case "${RECORD_VIDEO}" in
  0|1) ;;
  *) echo "[ERROR] RECORD_VIDEO must be 0 or 1 (got '${RECORD_VIDEO}')." >&2; exit 1 ;;
esac

is_positive_int "${VIDEO_LENGTH}" || { echo "[ERROR] VIDEO_LENGTH must be a positive integer (got '${VIDEO_LENGTH}')." >&2; exit 1; }
is_positive_int "${NUM_SAMPLES}"  || { echo "[ERROR] NUM_SAMPLES must be a positive integer (got '${NUM_SAMPLES}')." >&2; exit 1; }
is_non_negative_int "${SAMPLE_SEED}" || { echo "[ERROR] SAMPLE_SEED must be a non-negative integer (got '${SAMPLE_SEED}')." >&2; exit 1; }

if [[ -n "${MAX_ITERATIONS}" ]] && ! is_positive_int "${MAX_ITERATIONS}"; then
  echo "[ERROR] MAX_ITERATIONS must be a positive integer when set (got '${MAX_ITERATIONS}')." >&2
  exit 1
fi

for LR in ${REWARD_LRS}; do
  is_positive_float "${LR}" || { echo "[ERROR] REWARD_LRS entries must be positive (got '${LR}')." >&2; exit 1; }
done
for REG in ${REG_STRENGTHS}; do
  is_non_negative_float "${REG}" || { echo "[ERROR] REG_STRENGTHS entries must be >= 0 (got '${REG}')." >&2; exit 1; }
done
for R in ${REWARD_UPDATES}; do
  is_positive_int "${R}" || { echo "[ERROR] REWARD_UPDATES entries must be positive ints (got '${R}')." >&2; exit 1; }
done
for P in ${POLICY_UPDATES}; do
  is_positive_int "${P}" || { echo "[ERROR] POLICY_UPDATES entries must be positive ints (got '${P}')." >&2; exit 1; }
done

for SEED in ${SEEDS}; do
  is_non_negative_int "${SEED}" || { echo "[ERROR] SEEDS entries must be non-negative integers (got '${SEED}')." >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# Sampling: build the Cartesian grid, shuffle deterministically, take N.
# Each emitted line is: reward_lr\treg_strength\treward_updates\tpolicy_updates
# ---------------------------------------------------------------------------
export REWARD_LRS REG_STRENGTHS REWARD_UPDATES POLICY_UPDATES NUM_SAMPLES SAMPLE_SEED

mapfile -t SAMPLES < <(python - <<'PY'
import os, random, itertools, sys
lrs   = os.environ["REWARD_LRS"].split()
regs  = os.environ["REG_STRENGTHS"].split()
rupd  = os.environ["REWARD_UPDATES"].split()
pupd  = os.environ["POLICY_UPDATES"].split()
grid  = list(itertools.product(lrs, regs, rupd, pupd))
n_req = int(os.environ["NUM_SAMPLES"])
if not grid:
    sys.exit("[ERROR] Hyperparameter grid is empty.")
random.Random(int(os.environ["SAMPLE_SEED"])).shuffle(grid)
for t in grid[:n_req]:
    print("\t".join(t))
PY
)

NUM_DRAWN=${#SAMPLES[@]}
NUM_SEEDS=$(awk -v s="${SEEDS}" 'BEGIN { n = split(s, _, /[[:space:]]+/); print n }')
TOTAL_RUNS=$((NUM_DRAWN * NUM_SEEDS))

if [[ ${NUM_DRAWN} -lt ${NUM_SAMPLES} ]]; then
  echo "[INFO] Requested ${NUM_SAMPLES} samples but grid only has ${NUM_DRAWN}; using ${NUM_DRAWN}." >&2
fi

# ---------------------------------------------------------------------------
# Manifest CSV (idx -> hyperparams). Written before any training so it
# survives a crashed sweep.
# ---------------------------------------------------------------------------
LOG_PARENT="logs/irl/${EXPERIMENT_NAME}"
mkdir -p "${LOG_PARENT}"
MANIFEST="${LOG_PARENT}/sweep_manifest.csv"
{
  echo "sample_idx,reward_lr,reg_strength,reward_updates,policy_updates,sample_seed,seeds"
  for IDX in "${!SAMPLES[@]}"; do
    IFS=$'\t' read -r LR REG R_UPD P_UPD <<<"${SAMPLES[$IDX]}"
    printf "%02d,%s,%s,%s,%s,%s,%s\n" "${IDX}" "${LR}" "${REG}" "${R_UPD}" "${P_UPD}" "${SAMPLE_SEED}" "${SEEDS// /;}"
  done
} >"${MANIFEST}"

# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
echo "================================================================="
echo " IRL random hyperparameter sweep  (alpha=0)"
echo "================================================================="
echo " task             : ${TASK}"
echo " expert data      : ${EXPERT_DATA_PATH}"
echo " reward_lrs       : ${REWARD_LRS}"
echo " reg_strengths    : ${REG_STRENGTHS}"
echo " reward_updates   : ${REWARD_UPDATES}"
echo " policy_updates   : ${POLICY_UPDATES}"
echo " grid size        : $(awk -v lr="${REWARD_LRS}" -v r="${REG_STRENGTHS}" -v ru="${REWARD_UPDATES}" -v pu="${POLICY_UPDATES}" 'BEGIN {
  nl=split(lr,_,/[[:space:]]+/); nr=split(r,_,/[[:space:]]+/);
  nru=split(ru,_,/[[:space:]]+/); npu=split(pu,_,/[[:space:]]+/);
  print nl*nr*nru*npu }')"
echo " samples drawn    : ${NUM_DRAWN}"
echo " sample_seed      : ${SAMPLE_SEED}"
echo " training seeds   : ${SEEDS}"
echo " logger           : ${LOGGER}"
if [[ "${LOGGER}" == "wandb" ]]; then
  echo " log project      : ${LOG_PROJECT_NAME}"
fi
echo " experiment name  : ${EXPERIMENT_NAME}"
if [[ -n "${MAX_ITERATIONS}" ]]; then
  echo " max iterations   : ${MAX_ITERATIONS}"
fi
if [[ "${RECORD_VIDEO}" == "1" ]]; then
  echo " final video      : ${VIDEO_LENGTH} env steps (~$(awk "BEGIN{printf \"%.1f\", ${VIDEO_LENGTH}/50}") s at 50 Hz)"
else
  echo " final video      : disabled (RECORD_VIDEO=0)"
fi
echo " extra args base  : ${BASE_EXTRA_ARGS}"
echo " total runs       : ${TOTAL_RUNS}"
echo " manifest         : ${MANIFEST}"
echo "-----------------------------------------------------------------"
printf " %-4s %-12s %-12s %-6s %-6s\n" "idx" "reward_lr" "reg_str" "r_upd" "p_upd"
for IDX in "${!SAMPLES[@]}"; do
  IFS=$'\t' read -r LR REG R_UPD P_UPD <<<"${SAMPLES[$IDX]}"
  printf " %-4s %-12s %-12s %-6s %-6s\n" "${IDX}" "${LR}" "${REG}" "${R_UPD}" "${P_UPD}"
done
echo "================================================================="
echo ""

# ---------------------------------------------------------------------------
# Sweep loop.
# ---------------------------------------------------------------------------
RUN_IDX=0
FAILED_RUNS=()

for IDX in "${!SAMPLES[@]}"; do
  IFS=$'\t' read -r LR REG R_UPD P_UPD <<<"${SAMPLES[$IDX]}"
  IDX_TAG=$(printf "%02d" "${IDX}")

  for SEED in ${SEEDS}; do
    RUN_IDX=$((RUN_IDX + 1))
    RUN_NAME="irl_s${IDX_TAG}_seed_${SEED}"

    echo "-----------------------------------------------------------------"
    echo " [${RUN_IDX}/${TOTAL_RUNS}] sample=${IDX_TAG}  seed=${SEED}"
    echo "    reward_lr=${LR}  reg=${REG}  r_upd=${R_UPD}  p_upd=${P_UPD}"
    echo "-----------------------------------------------------------------"

    CMD=(
      "${PYTHON}" scripts/irl/train_irl.py
      --task "${TASK}"
      --expert_data_path "${EXPERT_DATA_PATH}"
      --bc_weight 0
      --reward_learning_rate "${LR}"
      --reward_regularization_strength "${REG}"
      --reward_updates_per_cycle "${R_UPD}"
      --rl_updates_per_cycle "${P_UPD}"
      --seed "${SEED}"
      --experiment_name "${EXPERIMENT_NAME}"
      --run_name "${RUN_NAME}"
      --logger "${LOGGER}"
    )
    if [[ "${LOGGER}" == "wandb" ]]; then
      CMD+=(--log_project_name "${LOG_PROJECT_NAME}")
    fi
    if [[ -n "${MAX_ITERATIONS}" ]]; then
      CMD+=(--max_iterations "${MAX_ITERATIONS}")
    fi
    if [[ "${RECORD_VIDEO}" == "1" ]]; then
      CMD+=(--video --video_final_only --video_length "${VIDEO_LENGTH}")
    fi
    # shellcheck disable=SC2086
    CMD+=(${BASE_EXTRA_ARGS})

    echo "+ ${CMD[*]}"

    # Snapshot the experiment log dir before the run so we can pick out the
    # new run subfolder regardless of the timestamped name train_irl.py picks.
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
      echo "[WARN] Run sample=${IDX_TAG} seed=${SEED} failed; continuing sweep." >&2
      FAILED_RUNS+=("sample=${IDX_TAG} seed=${SEED} (lr=${LR} reg=${REG} r=${R_UPD} p=${P_UPD})")
    fi
    echo ""
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
echo " Logs     : ${LOG_PARENT}/"
echo " Manifest : ${MANIFEST}"
echo " View     : wandb dashboard for project ${LOG_PROJECT_NAME} (when LOGGER=wandb)"
echo "================================================================="

[[ ${#FAILED_RUNS[@]} -eq 0 ]]
