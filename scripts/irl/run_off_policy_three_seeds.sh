#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "sac" && "$1" != "pmd" ) ]]; then
  echo "Usage: $0 {sac|pmd} [extra train args...]" >&2
  exit 2
fi

algorithm=$1
shift
isaac_python="${ISAAC_IRL_PYTHON:-/home/andreas/miniconda3/envs/isaac_irl/bin/python}"

for seed in 1 2 3; do
  "$isaac_python" scripts/irl/train_irl.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --algorithm "$algorithm" \
    --seed "$seed" \
    --num_envs 4096 \
    --max_iterations 500 \
    --bc_weight 0 \
    --use_ground_truth_reward_weights \
    --validation_interval 25 \
    --video --video_final_only --video_length 250 \
    "$@"
done
