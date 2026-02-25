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

### Train IRL + Record Video Every 50 Iterations
```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5 \
  --video \
  --video_interval_iterations 50 \
  --video_length 200
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

### TensorBoard
```bash
tensorboard --logdir logs/irl --port 6006
```

## End-to-End Example
```bash
python scripts/rsl_rl/train_rl.py --task Isaac-Lift-Cube-Franka-v0 --headless
python scripts/rsl_rl/play.py --task Isaac-Lift-Cube-Franka-v0 --headless --load_run "<run>" --checkpoint "<ckpt>"
python scripts/recording/record_synthetic_demos.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_demos 1000 --load_run "<run>" --checkpoint "<ckpt>"
python scripts/irl/train_irl.py --task Isaac-Lift-Cube-Franka-v0 --headless --expert_data_path logs/demos/franka_lift/demos.hdf5
```

## Tests
```bash
PYTHONPATH=src pytest -q
```
