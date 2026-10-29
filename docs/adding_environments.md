# Adding an environment

The policy/reward learners are reusable across tasks; their observation,
action, and feature dimensions are inferred at construction. For a compatible
**Isaac Lab manager-based, single-agent task with vector observations and
continuous actions**, adding an experiment is usually a configuration change
plus the checks below. It is not yet a universal Gym environment adapter.

This guide describes the code as checked on 2026-09-19. A new environment
still needs its own action-range, feature, termination, and performance
validation. A successful run on Lift does not establish those contracts for
another task.

## 1. Check whether the task fits the existing integration

| Requirement | What the current code expects |
| --- | --- |
| Task registration | A Gym task ID registered when `isaaclab_tasks` is imported, with an Isaac `env_cfg_entry_point` usable by `parse_env_cfg`. |
| Environment boundary | Isaac's rsl_rl wrapper, then `RslRlVecEnvAdapter`; exposes `step`, `reset`, `get_observations`, `num_envs`, `action_dim`, `max_episode_length`, `device`, and `unwrapped`. |
| Observations | Tensor groups suitable for an MLP. PMD prefers `actor` then `policy`; critic inputs prefer `critic`, `policy`, then `actor`. Standard Isaac/rsl_rl observation containers are supported; arbitrary Gym dictionaries are not automatically a working pipeline. |
| Actions | PMD produces `a = b * tanh(z)` with **one scalar symmetric bound** `b` for every coordinate. The chosen native action interface must fit that contract. |
| Reward features | A reward manager exposing named scalar terms. Features are raw, unweighted term values, collected before auto-reset. Nonzero-weight terms are selected by default. |
| PMD terminal handling | Reward, observation, and termination managers; `extras["time_outs"]` or `unwrapped.reset_time_outs`; true pre-reset successor observations for timeouts. |
| Successor backend | Reward must be explicitly linear in the selected features. A nonlinear learned reward requires direct Q or PPO. |

`N` denotes parallel environments, `O` observation width, `A` action width,
and `D` feature width. Check finite tensors with observations `[N,O]`,
actions `[N,A]`, features `[N,D]`, and reward/done masks `[N]`. Actor and
critic may have different observation widths; you do not hard-code these or
`D` in the learner.

Important limits:

- **DirectRLEnv without a reward manager is not configuration-only**, even
  for PPO with environment reward: the pipeline always builds a manager-based
  feature source. It needs another feature/capture integration.
- Images, recurrent/history-dependent observations, discrete actions,
  multi-agent tasks, and asymmetric/per-coordinate action transforms need
  explicit support; do not assume flattening or a large scalar bound solves
  those requirements.
- The pre-reset observation tap performs an extra observation computation.
  Noise, history mutation, or commands changing between that pass and the
  normal pass can invalidate it. The collector checks surviving environments
  for a mismatch. Use a side-effect-free terminal capture path when needed;
  do not remove the check to make a task run.

For custom task packages, add their registration import **after AppLauncher
starts** in both training and recording dependency loading, or a shared
runtime registration helper called by both. Do not move Isaac imports to
module scope. Verify the actual registry inside a launched app rather than
guessing task IDs.

## 2. Create a task-local configuration

Use `configs/<task-slug>/experiment.yaml`. The loader lowercases the task ID,
removes the `Isaac-` and recognized `-v0`/`-v1`/`-v2` tokens, and replaces
hyphens with underscores. For example, `Isaac-Reach-Franka-v0` maps to
`reach_franka`. For unusual IDs, check `config.task_slug` or pass `--config`
explicitly; do not assume every version suffix is stripped.

The following is a **bring-up template**, not tuned training settings.
Replace the placeholder ID/name, then review the action bound and initial
spread in step 3 before running it. All omitted fields use schema defaults.

```yaml
experiment_name: your_task
logger: noop
seed: 42
device: cuda:0
max_iterations: 20

env:
  name: Isaac-Your-Task-v0
  device: cuda:0
  num_envs: 32
  add_success_bonus: false

feature_map:
  ignored_reward_terms: []
  force_include_terms: []

reward:
  source: environment
  fit_diagnostic: false
  model: {is_linear: true}
bc: {weight: 0.0}

algo:
  type: pmd
  ppo: {}                    # set the new task's PPO tuning here
  pmd:
    policy_improvements_per_batch: 1
    actor_batch_size: 128
    train:
      episodes_per_batch: 32
      critic_batch_size: 64
      critic_updates_per_batch: 64
      initial_critic_updates: 32
      replay: {learning_starts: 1024, capacity: 100000}
    policy:
      hidden_dims: [128, 128]
      action_bound: 1.0      # provisional: audit native action units first
      initial_std: 1.0       # provisional: audit executed exploration separately
    value:
      type: successor
      num_heads: 3
      width: 256
      residual_blocks: 1
      successor: {latent_dim: 64, head_hidden_dim: 64, control_quantile: 0.25}
      direct: {target_subset_size: 2, control_reduction: min}

actor_critic:
  actor_hidden_dims: [128, 128]
  critic_hidden_dims: [128, 128]
  activation: elu

runner:
  validation_interval: 0
  validation_interval_transitions: null
  transition_budget: null
  imitator_source: collected
```

Replace `algo.ppo` and `actor_critic` with the new task's baseline settings
for meaningful PPO training. They are **not automatically imported** from
Isaac's task-specific PPO configuration. `actor_critic` affects PPO only;
PMD's hidden widths live in `algo.pmd.policy`. Its activation is currently
ELU in code, not a YAML-selectable setting. Keep the critic architecture
fixed initially so action-interface tests do not also become critic ablations.

Unknown keys are rejected. CLI overrides must target the selected nested
block, e.g. `--set algo.pmd.policy.action_bound=2.0`, not a competing
`algo.policy` block. Native Isaac settings such as joint action scale,
observation corruption, early terminations, and episode duration are **not**
arbitrary fields in this repo's `EnvCfg`. Set them in the registered task's
config or an explicit shared environment-config builder used by training
and recording.

## 3. Audit actions, features, and task metrics manually

Before learning, write down what one native action does: torque, velocity,
position relative to a default pose, or an increment from the current pose.
Check every scale/offset/clip between the actor and actuator. A generic
`Box` declaration alone is not sufficient evidence of a useful range.

Choose a command range that permits the task, then choose initial exploration
within it. Use a scripted controller or a working baseline on matched starts
to test the range, including contacts/grasping when relevant. Inspect actions
per coordinate and resulting motion. Do not copy Lift's bound of 4 into an
unrelated task, or assume `initial_std=1` means native-action std 1 after tanh
and scaling. See [the Lift rationale](lift_action_bounds.md) for that distinction
and a concrete diagnostic. `Action/*` currently pools action coordinates;
separate joint/gripper plots require a manual diagnostic or added logging.

Verify the feature list printed by Isaac's reward manager:

- Keep all nonzero-weight reward terms for an environment-reward PMD smoke
  test. The fixed scorer checks that `step_dt * dot(weights, features)`
  reconstructs the returned reward; ignoring a contributing term breaks that
  contract. A reward-transforming wrapper can break it too.
- Include zero-weight terms only deliberately with `force_include_terms`.
  Confirm they exist and return finite scalar-per-environment values.
- `env.reward_curriculum=false` removes known `modify_reward_weight` terms;
  recording always freezes these weights. Environment-reward PPO can opt in
  with `env.reward_curriculum=true`. PMD/IRL currently require fixed weights;
  other weight-changing mechanisms are not handled and the tap rejects them.
  Record which reward definition is used. See [curriculum rationale](episode_training.md#preserve-standard-ppo-and-make-curriculum-explicit).

**Success reporting is still task-specific.** The current factory attempts
the Lift `object_reached_goal` predicate, not a general success registry.
For a non-Lift task, set `env.add_success_bonus: false` and remove
`success_bonus` from `force_include_terms`. This disables feature injection,
not success-function selection. An incompatible predicate normally logs a
warning and disables success reporting; if it happens to run, verify that its
meaning actually matches the new task. Return-based validation remains usable.
To add meaningful task success, implement a predicate and an explicit task
dispatch in the success factory shared by training and recording. Do not
interpret a missing success metric as a zero success rate.

## 4. Run a small integration check, including actual updates and resets

Use the Python environment containing Isaac Lab. Set these shell variables
to the task/config you just created; they are placeholders, not a registered
example environment:

```bash
task_id="Isaac-Your-Task-v0"
task_config="configs/your_task/experiment.yaml"

ruff check . && ruff format --check . && python -m pytest -q

python scripts/train.py --task "$task_id" --config "$task_config" \
  --headless --logger noop --reward_source environment \
  --algorithm pmd --value_backend successor --max_iterations 20

python scripts/train.py --task "$task_id" --config "$task_config" \
  --headless --logger noop --reward_source environment \
  --algorithm pmd --value_backend direct --max_iterations 20

python scripts/train.py --task "$task_id" --config "$task_config" \
  --headless --logger noop --reward_source environment \
  --algorithm ppo --max_iterations 5
```

With the template, PMD collects 32 complete episodes per cycle. At a 250-step
horizon without early endings, that is 8,000 transitions, enough to reach
`learning_starts=1024` and project in the first cycle after warm-up. Early
failures can reduce that count. Confirm nonzero `policy_updates`, finite
critic/projection metrics, and a checkpoint under
`logs/train/<experiment_name>/<run>/model_*.pt`. A short run that only fills
replay is not an update test. These runs establish integration, not learning.

Also exercise a **real time-limit reset through PMD collection** and, where
applicable, a genuine failure termination. Every PMD batch now finishes its
selected episodes. Inspect `Data/truncated_fraction`, `Data/terminated_fraction`, and
`Data/bootstrap_disabled_fraction` with logging or a diagnostic. The console
does not print every `Data/*` value; `--logger noop` retains only its normal
console summaries and checkpoints.

For truncations, the successor observation must be the state before auto-reset
and bootstrapping must remain enabled. For genuine terminations it must be
disabled. Missing timeout information or a terminal-observation mismatch is
an integration problem to resolve, not a reason to substitute all-false masks.
PPO may stagger initial episode ages; PMD does not. A PMD episode must end via
the environment's own flags, within `max_episode_length`, or collection raises.

Enable PMD validation separately, using complete episodes:

```bash
python scripts/train.py --task "$task_id" --config "$task_config" \
  --headless --logger noop --reward_source environment --algorithm pmd \
  --max_iterations 6 --set runner.validation_interval=3 \
  --set runner.validation_interval_transitions=null \
  --set runner.validation_episodes=32
```

Check the validation return, task-appropriate success if installed, and both
deterministic and stochastic policy evaluation. Then train against the fixed
environment reward with a meaningful budget and repeated seeds. Only after
that baseline works should you introduce reward learning or BC. Compare
algorithms by transitions and wall-clock time, not iteration numbers alone.

Changing `num_envs` changes collection parallelism, not the configured number
of critic steps or policy improvements. Choose episodes and compute separately,
and inspect unused-lane overhead. See
[the PMD cadence equations](pmd_updates.md#4-update-order-and-target-synchronization),
and [episode-budget accounting](episode_training.md#count-the-interactions-actually-paid-for).

## 5. Record compatible demonstrations, then enable IRL

Keep a **loadable expert config** containing the algorithm and actor settings
that actually trained the checkpoint. For a PPO expert, create
`configs/your_task/expert.yaml` from the task config with `algo.type: ppo`,
`reward.source: environment`, and the exact PPO architecture/normalization.
Bake any relevant training CLI overrides into it.

The recorder has no `--algorithm` override: training with `--algorithm ppo`
does not change the original YAML, and handing that still-PMD YAML to the
recorder reconstructs the wrong actor. The saved run's
`params/experiment.yaml` is a diagnostic dump, **not** a directly reloadable
experiment file: it contains `algorithm` and an environment summary rather
than the launcher's `algo`/`env` structure.

For a non-Lift task with no injected feature-only terms:

```bash
python scripts/record_demos.py --task "$task_id" \
  --config configs/your_task/expert.yaml --headless \
  --checkpoint logs/train/your_task/EXPERT_RUN/model_EXPERT_ITER.pt \
  --num_envs 32 --num_demos 32 --no-video --no-add_success_bonus \
  --output_dir logs/demos/your_task/bringup
```

Replace the checkpoint placeholders with a verified expert. Use a fresh
output directory: the recorder overwrites an existing `demos.hdf5` there.
Its `--config` reconstructs the actor; it does **not** automatically apply all
training environment/feature choices. In particular:

- `--add_success_bonus` defaults to true in the recorder, even if the training
  YAML disables it. Pass `--no-add_success_bonus` explicitly for other tasks.
- Match ignored terms with `--ignored_reward_terms name_a,name_b` if needed.
  Arbitrary zero-weight forced terms beyond `success_bonus` are not exposed
  by the recorder CLI; supporting them requires a recording configuration
  change, not merely changing the training YAML.
- Apply custom environment modifications in both construction paths. Match
  observation/action semantics, physics/control timestep, feature definitions,
  and reset distribution. Keep the original bound when recording an old PMD
  checkpoint; changing it changes behavior without changing weight shapes.
- Default `demo_length` is the environment horizon. A shorter `--demo_length`
  splits recordings into segments; it does **not** change the environment's
  horizon or force a physical reset at each segment boundary.

Inspect a few episodes before recording a large dataset. Features must match
the online names and meanings. HDF5 metadata must include the version,
pre-reset capture timing, and raw-unweighted value semantics from
[feature_semantics.py](../src/data/feature_semantics.py). Do not relabel a
legacy dataset to bypass this check; regenerate it when its semantics differ.

Smoke-test IRL with those fresh demos:

```bash
python scripts/train.py --task "$task_id" --config "$task_config" \
  --headless --logger noop --reward_source learned --algorithm pmd \
  --expert_data_path logs/demos/your_task/bringup/demos.hdf5 \
  --max_iterations 6 --set runner.imitator_source=collected
```

Check expert loading, completed imitator episodes, reward updates, and finite
feature-gap metrics. With successor values, keep `reward.model.is_linear:
true`. Use comparable expert/imitator episode, discount, reset, and horizon
semantics before interpreting occupancy matching. The current code does not
automatically disable early terminations for IRL; time limits still bootstrap
in TD learning. Changing that objective needs an explicit design decision.

Keep BC and demonstration replay seeding off initially. If later enabling BC
with a PMD actor, verify every expert action fits its bound; an unbounded PPO
expert is not automatically compatible. Do not silently clip the dataset.

## 6. Where extensions belong

| New requirement | Boundary to extend |
| --- | --- |
| Custom task registration / environment configuration | Shared helpers in `src/training/`, called by both `pipeline.py` and `record.py` after app startup |
| Different environment step/reset/terminal contract | `src/adapters/`, with tests for timeout and genuine-terminal transitions |
| New feature definitions | `src/reward/`; features consume the env and reward models consume feature tensors |
| Different feature capture timing | An adapter capture source, not reward-model or TD-target special cases |
| Task-specific success | A predicate plus explicit factory selection used by training and recording |
| New configurable choice | Validated dataclass in `src/config/schema.py`; add CLI exposure only when needed |
| Per-coordinate/asymmetric action transform | Actor density/inverse, collection validation, diagnostics, BC validation, and recording must agree; not an env-only clipping patch |

Keep launch scripts thin and algorithms unaware of the simulator. Add focused
tests beside `test_training_pipeline.py`, `test_off_policy_stack.py`,
`test_pre_reset_tap.py`, `test_transition_collector.py`, or
`test_demo_writer.py` as appropriate. Do not weaken an existing data contract
to accommodate a new task silently.

Before calling the port ready, retain its YAML, resolved run settings, manual
action/termination checks, checkpoint, demonstration metadata, and validation
results. Write down any task-specific code and untested assumptions. The goal
is one explicit task configuration and small boundary extensions where
needed, not a second training loop.
