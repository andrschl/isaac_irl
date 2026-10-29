# isaac_irl

Learn a reward from demonstrations, train a policy against it, optionally
regularize that policy toward the demonstrations.

## Setup

Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) first, then:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Put `WANDB_API_KEY` in `.env` (see `.env.example`), or run with `--logger noop`.

## Running an experiment

`--task` picks the Isaac task *and* its config: `configs/<task-slug>/experiment.yaml`.
Runs land in `logs/train/<experiment_name>/<timestamp>_<run_name>/`.

**1. Expert policy** — PPO on the environment's own reward:

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/ppo.yaml
```

**2. Demonstrations** — roll that checkpoint out, writing
`logs/demos/<task-slug>/demos.hdf5`. `--config` is the config of the experiment
the demonstrations are *for*, so they are recorded on that task, with its
features and its episode; `--teacher_config` is the one the checkpoint's actor
architecture comes from. Only complete successful episodes are kept, and an
existing file is never overwritten without `--overwrite`:

```bash
python scripts/record_demos.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/experiment.yaml \
  --teacher_config configs/lift_cube_franka/ppo.yaml \
  --checkpoint logs/train/lift_cube_franka_ppo/<run>/model_1500.pt --num_demos 10
```

**3. IRL** — learn the reward and train against it:

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --expert_data_path logs/demos/lift_cube_franka/demos.hdf5
```

**4. Offline BC** — fit the off-policy actor to the demonstrations with no
simulator at all, so several online arms can start from one actor:

```bash
python scripts/train_bc.py --task Isaac-Lift-Cube-Franka-v0 \
  --expert_data_path logs/demos/lift_cube_franka/demos.hdf5 \
  --output logs/bc/actor_s42.pt
python scripts/train.py --task ... --init_policy_checkpoint logs/bc/actor_s42.pt
```

Smoke-test any of this on cartpole (seconds per iteration, no Franka):

```bash
python scripts/train.py --task Isaac-Cartpole-v0 --headless --logger noop --max_iterations 20
```

## Varying a run

| | |
|---|---|
| `--algorithm ppo\|pmd` | on-policy baseline, or off-policy policy mirror descent |
| `--value_backend direct\|successor` | PMD's action-value: scalar Q, or successor features |
| `--reward_source learned\|environment\|environment_plus_learned` | maximize the learned reward, the task's own, or their sum |
| `--fit_diagnostic` | with `environment`, fit the reward model anyway, to compare |
| `--bc_weight 0.3` | `L = 0.7 L_RL + 0.3 L_BC`; `1.0` is pure BC, no env stepping |
| `--expert_num_trajectories 50` | demonstration-count ablation |
| `--seed 1 --num_envs 4096` | |
| `--max_iterations N` / `--transition_budget N` | stop by iterations, or retained training transitions |
| `--set runner.total_transition_budget=N` | hard budget including evaluation and unused simulator lanes |

Anything else in the config is reachable by path, and a key that does not exist
is an error rather than a silently ignored line:

```bash
--set algo.pmd.train.episodes_per_batch=32 \
--set algo.pmd.train.critic_updates_per_batch=1000 \
--set algo.pmd.actor_projection_kl=0.02 \
--set runner.reward_updates_per_cycle=8
```

Each algorithm's tuning lives in a block named after it, so switching arms
selects that arm's settings instead of falling back to defaults — a benchmark
where one arm is tuned and the other is not measures the tuning:

```yaml
algo:
  type: pmd
  ppo: {learning_rate: 1.0e-3, gamma: 0.99}
  pmd:
    policy_improvements_per_batch: 1
    value:
      type: successor
      num_heads: 5              # shared by both backends
      successor: {latent_dim: 128}
      direct: {target_subset_size: 2}
```

## Batch IRL on a sparse reward

`configs/lift_cube_franka/{sparse_bc,sparse_irl_bc,irl_bc}.yaml` are the three
online arms of a study that asks whether a reward fit to ten demonstrations
beats a sparse terminal reward per environment interaction. They differ from
each other by `reward.source` and nothing else. The task, the equations, the
data and compute accounting, and the launch commands are in the
[batch IRL protocol](docs/batch_irl_protocol.md).

## Benchmarking against the task's own reward

`--reward_source environment` disables reward learning. PMD collects full
episode batches; native PPO retains short rollouts and its reward curriculum.
Use the dedicated PPO config, not just `--algorithm ppo` on PMD's data/budget
settings. For fair comparisons, distinguish native task performance from a
matched fixed-reward/sample-budget ablation. See [episode training and
benchmarking](docs/episode_training.md) for the protocol and accounting.

```bash
# cartpole: both PMD value representations
for backend in successor direct; do
  python scripts/train.py --task Isaac-Cartpole-v0 --headless \
    --reward_source environment --algorithm pmd \
    --value_backend "$backend" --run_name "bench-pmd-$backend"
done

# lift: the off-policy arms are tuned at 32 envs, the PPO baseline at 4096
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --reward_source environment --value_backend successor
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/ppo.yaml
```

Historical cartpole measurements under the **old short-chunk protocol**
(150k training-transition budget, per-step return, theoretical maximum ≈ 0.0167).
These are not results for the new episode protocol; backend selection also
needs re-verification following the nested CLI override fix:

| arm | best return/step | transitions to near-optimal |
|---|---|---|
| PMD + successor Q | 0.01654 | 8,192 |
| PMD + direct Q | 0.01643 | 61,440 |
| PPO | 0.01580 | 139,264 |

## Reading a run

For external checkpoint comparisons, use the [evaluation dashboard](docs/evaluation_dashboard.md):
one plot overlays all algorithms against actual environment interactions.

For single-GPU hyperparameter experiments, see [sequential PMD tuning](docs/sequential_tuning.md):
fixed interaction budgets, uninterrupted training, and automatic seed replication within a wall-clock cap.
For the longer follow-up with two concurrent learners, see [parallel PMD tuning](docs/parallel_tuning.md).

Every run emits `Summary/*`. W&B uses `Summary/environment_transitions` as
the default x-axis: all training/evaluation/unused-lane interactions.
`Summary/transitions` retains the training-only count. The console shows both:

```
[INFO] iter 42  98,304 transitions  feature_gap=0.655  reward_loss=-1.42  reward_cos=0.809
                critic_loss=0.312  q_mean=4.81  policy_std=0.44  actor_kl=0.0183
[INFO] validation 42  success=0.62  success_stoch=0.48  success_expert=0.93  return/step=1.84
```

| Question | Keys |
|---|---|
| Is the policy working? | `success_rate`, `success_rate_stochastic`, `expert_success_rate`, `return_per_step`, `best_*`, `transitions_to_success_rate_<t>` |
| Is IRL working? | `feature_gap_norm` (what it minimizes), `reward_weight_cosine` (learned reward vs. the task's own weights), `reward_loss` |
| Is RL healthy? | `critic_loss`, `q_mean`, `policy_std`, `actor_kl`, `policy_updates`, `train_reward_per_step` |
| Is BC contributing? | `bc_loss`, `bc_val_loss` — only when `bc.weight > 0` |

Everything else stays in `PMD/*`, `DirectQ/*`, `SuccessorQ/*`, `Data/*`,
`Action/*`, `Reward/*`, `ValidationFeatures/*` for debugging.

## Also

```bash
# resume
python scripts/train.py --task ... --resume --load_run <run> --checkpoint "model_.*.pt"
# video: clips during training, or only the final rollouts
python scripts/train.py --task ... --video --video_interval_iterations 50
python scripts/train.py --task ... --video --video_final_only --video_length 250
# tests
pytest -q
```

See [episode collection, validation, budgets, and curriculum](docs/episode_training.md),
the [algorithm overview](docs/algorithms.md), the exact
[PMD policy and value updates](docs/pmd_updates.md), and the
[Lift action-bound and architecture rationale](docs/lift_action_bounds.md)
(including the comparison with Raffin's SAC article). The config schema is
[src/config/schema.py](src/config/schema.py).

To port another task, follow [Adding an environment](docs/adding_environments.md):
supported contracts, a config template, simulator checks, demonstration
recording, and the places that still need task-specific integration.
