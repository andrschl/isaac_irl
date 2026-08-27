# isaac_irl
IRL workflow for Isaac Lab: train RL policies, collect synthetic demos, and train a learned reward.

## Setup
Install Isaac Lab first: https://isaac-sim.github.io/IsaacLab/

Use your existing Isaac Lab environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Or create a local Conda environment:

```bash
conda env create -f environment.yml
conda activate isaac_irl
```

## Key Commands
Run all commands from repository root.

### Train RL Policy
```bash
python scripts/rsl_rl/train_rl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless
```

### Evaluate / Export Policy
```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --load_run "<run_folder_name>" \
  --checkpoint "model_2000.pt"
```

Livestream instead of headless: replace `--headless` with `--livestream 2` and
connect via the Isaac Sim WebRTC Streaming Client at `<host-ip>:8211`.

### Collect Synthetic Demos
```bash
python scripts/recording/record_synthetic_demos.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --num_demos 1000 \
  --load_run "<run_folder_name>" \
  --checkpoint "model_2000.pt"
```

By default, demo rollout length matches the environment's built-in max episode length.
Use `--demo_length <steps>` only if you want to override it.

### Train IRL
```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5
```

Optional IRL discount override:
- `--irl_discount_gamma <0..1>` (default: uses PPO `gamma`)
- IRL return targets are normalized by episode length by default (`irl.normalize_returns_by_episode_length: true`).

### Mix Behavioral Cloning into the Policy Loss

BC is an independent regularizer for PPO, SAC, and PMD:

```
L_actor = (1 - bc_weight) * L_RL + bc_weight * L_BC
```

Set `bc.weight` in `configs/franka_lift/experiment.yaml` or pass
`--bc_weight`. The dataset is the same HDF5 written by
`record_synthetic_demos.py`.

```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5 \
  --bc_weight 0.3
```

- `--bc_weight 0`: RL only; no BC loader, batch, or gradient is constructed.
- `--bc_weight 1`: pure offline BC.
- `--bc_loss_type nll|mse`: likelihood or mean-action regression.

Use `scripts/irl/bc_weight_lr_sweep.sh` for the BC-weight/LR sweep. Its
`BC_WEIGHTS`, `LR_MULTIPLIERS`, and `SEEDS` environment variables control
the grid. Exact SAC, PMD, value-backend, and UTD semantics are documented in
[docs/rl_algorithms.md](docs/rl_algorithms.md).

### Compare RL Algorithms on Fixed-Reward Cartpole
```bash
python scripts/rl/cartpole_benchmark.py --algorithm pmd --value_backend direct --headless
ISAAC_IRL_PYTHON=python scripts/rl/cartpole_benchmark.sh   # 5 variants x 3 seeds
python scripts/rl/create_wandb_overview.py --group cartpole-fixed-reward-v4
```

The benchmark trains against the environment's own reward (fixed, non-learnable,
checked every rollout against the reward features) and reports sample efficiency
in simulator transitions with deterministic 250-step validation. `Overview/*` is
the small cross-algorithm dashboard; per-algorithm diagnostics stay in their own
namespaces. `create_wandb_overview.py` takes the entity from `$WANDB_ENTITY`.

### Train IRL + Record Video Every 50 Iterations
```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5 \
  --video \
  --video_interval_iterations 50 \
  --video_length 1500
```

Videos are saved under `logs/irl/<experiment>/<run>/videos/train/`.
When `--video_interval_iterations` is used, video filenames use learning-iteration indices:
`rl-video-iter-0.mp4`, `rl-video-iter-50.mp4`, `rl-video-iter-100.mp4`, ...

### Resume IRL
```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5 \
  --resume \
  --load_run "<previous_irl_run_folder>" \
  --checkpoint "model_*.pt"
```

### Logging
IRL training logs to Weights & Biases by default. Use `--logger noop` for local smoke runs without external logging.

## End-to-End Example
```bash
python scripts/rsl_rl/train_rl.py --task Isaac-Lift-Cube-Franka-v0 --headless
python scripts/rsl_rl/play.py --task Isaac-Lift-Cube-Franka-v0 --headless --load_run "<run>" --checkpoint "<ckpt>"
python scripts/recording/record_synthetic_demos.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_demos 1000 --load_run "<run>" --checkpoint "<ckpt>"
python scripts/irl/train_irl.py --task Isaac-Lift-Cube-Franka-v0 --headless --expert_data_path logs/demos/franka_lift/demos.hdf5
```

## Tests
```bash
pytest -q
```
