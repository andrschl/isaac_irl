#!/usr/bin/env bash
# Fixed-reward Cartpole sample-efficiency benchmark: every variant, every seed.
set -euo pipefail

python_bin="${ISAAC_IRL_PYTHON:-python}"
project="${WANDB_PROJECT:-isaac-irl-cartpole-sample-efficiency}"
group="${WANDB_GROUP:-cartpole-fixed-reward-v4}"
budget="${TRANSITION_BUDGET:-500000}"
eval_interval="${EVAL_INTERVAL:-4096}"
num_envs="${NUM_ENVS:-256}"
seeds="${SEEDS:-1 2 3}"

# variant: <name> <algorithm> [extra args...]
variants=(
    "ppo|ppo|"
    "sac-direct|sac|--value_backend direct"
    "sac-successor|sac|--value_backend successor"
    "pmd-direct|pmd|--value_backend direct"
    "pmd-successor|pmd|--value_backend successor"
)

for variant in "${variants[@]}"; do
    IFS='|' read -r name algorithm extra <<< "${variant}"
    for seed in ${seeds}; do
        video_args=()
        if [[ "${seed}" == "1" ]]; then
            video_args+=(--video --video_length 250)
        fi
        "${python_bin}" scripts/rl/cartpole_benchmark.py \
            --algorithm "${algorithm}" \
            --run_name "${name}-seed-${seed}" \
            --seed "${seed}" \
            --num_envs "${num_envs}" \
            --transition_budget "${budget}" \
            --evaluation_interval_transitions "${eval_interval}" \
            --project "${project}" \
            --group "${group}" \
            --headless \
            ${extra} \
            "${video_args[@]}"
    done
done
