#!/usr/bin/env bash
# Fixed-reward sample-efficiency benchmark: every variant, every seed, one task.
set -euo pipefail

python_bin="${ISAAC_IRL_PYTHON:-python}"
task="${TASK:-cartpole}"
project="${WANDB_PROJECT:-isaac-irl-sample-efficiency}"
group="${WANDB_GROUP:-${task}-fixed-reward-v4}"
seeds="${SEEDS:-1 2 3}"
extra=("$@")

# variant: <name>|<algorithm>|<extra args>
variants=(
    "ppo|ppo|"
    "sac-direct|sac|--value_backend direct"
    "sac-successor|sac|--value_backend successor"
    "pmd-direct|pmd|--value_backend direct"
    "pmd-successor|pmd|--value_backend successor"
)

for variant in "${variants[@]}"; do
    IFS='|' read -r name algorithm variant_args <<< "${variant}"
    for seed in ${seeds}; do
        video_args=()
        if [[ "${seed}" == "1" ]]; then
            video_args+=(--video --video_length 250)
        fi
        "${python_bin}" scripts/rl/benchmark.py \
            --task "${task}" \
            --algorithm "${algorithm}" \
            --run_name "${name}-seed-${seed}" \
            --seed "${seed}" \
            --project "${project}" \
            --group "${group}" \
            --headless \
            ${variant_args} \
            "${video_args[@]}" \
            "${extra[@]}"
    done
done
