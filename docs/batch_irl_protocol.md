# Batch IRL on sparse-reward Lift

What the four arms are, what each one optimizes, and how to run them. The
configuration lives in `configs/lift_cube_franka/{sparse_bc,sparse_irl_bc,irl_bc}.yaml`;
this document is the part of the protocol that does not fit in a comment.

## The question

A sparse terminal reward tells a policy almost nothing: on Lift it arrives once
per episode, or never. The claim under test is that fitting a *dense* reward to
ten demonstrations, and maximizing that, recovers more of the task than the
sparse reward alone — and that it does so per environment interaction, not per
gradient step.

## Task

Easy Lift (`env.variant: lift_easy`), no reward curriculum, horizon `H = 250`
steps at `dt = 0.02 s`, and no discounting (`gamma = 1`). The episode genuinely
ends at `H`, so nothing bootstraps past it.

The environment pays one thing. Let `d_t` be the object-to-goal distance after
step `t` and `c_t = [d_t < 0.08 m]`. Then

```
g = 1   if  c_t = 1 for all t in {H-24, ..., H}  and no failure occurred
    0   otherwise
```

and the reward is `g` on the final transition and zero everywhere else.
Reaching the goal early pays nothing and ends nothing; only holding it to the
horizon does. A genuine failure (the object dropped) beats success on the step
where both would fire.

Every dense Lift reward term keeps its place in the reward manager at weight
zero. The environment stops paying for them, but they are still computed, so
they remain the feature vector

```
phi_t in R^6 = (reaching_object, lifting_object, object_goal_tracking,
                object_goal_tracking_fine_grained, action_rate, joint_vel)
```

read *before* Isaac's automatic reset, exactly as during training. The sparse
coordinate `terminal_success` is a seventh column of the collected feature
vector — the critics see it, because it is part of the reward being maximized —
but it is never part of `phi`, and its coefficient is never learned.

Because a finite-horizon value function is not stationary, the student's
observation carries two extra columns: remaining time `1 - t/H` and capped hold
progress `min(streak_t, 25) / 25`. Reading them never advances the task state.

## Arms

| Arm | Reward the policy maximizes | Reward learned |
| --- | --- | --- |
| BC | — (frozen after the offline fit) | no |
| Sparse RL + BC | `g` | no |
| Sparse RL + IRL + BC | `g + dt * w.phi` | `w` |
| IRL + BC | `dt * w.phi` | `w` |

The three online arms differ by one line of configuration (`reward.source`) and
nothing else, which `tests/test_config.py` asserts. Every arm starts from the
same behaviour-cloned actor.

## The reward learner

Under uniform sampling over complete attempted episodes, the feature
expectation of a policy `pi` is

```
mu(pi) = E_{tau ~ pi} [ sum_{t=1..T(tau)} dt * phi_t ]
```

with no discount, no division by episode length or by `H`, and no clipping.
Failed episodes count as they are, including one-step failures. Absorbing
continuation is zero and is represented by the existing masks, not by fake
replay rows.

The linear reward `r(phi) = dt * w.phi` is fit by descending

```
L(w) = w.mu(pi) - w.mu(expert) + lambda * ||w||^2
```

with RMSprop at `1e-4`, gradient clipping at 1.0, and a projection of `w` onto
the unit L2 ball after every step. The gradient is the measured feature gap
`mu(pi) - mu(expert)`, which is the quantity reported as
`IRL/feature_exp_diff_norm`.

The sparse coefficient is not in `w`. On the `environment_plus_learned` arm it
is held at the environment's own weight of one; on the `learned` arm it is held
at zero. Both are implemented as one fixed vector added to the learned one on a
shared named basis, so the critics always score a single 7-dimensional feature
vector.

## The cycle

One outer cycle is:

1. Freeze the behaviour policy. Collect 32 complete attempted episodes.
2. Add their real transitions to persistent replay, once.
3. Update the learned reward from those same episodes, then freeze it.
4. Run one block of PMD updates against the frozen reward.

Nothing inside either optimizer loop touches the simulator, and an update
without a preceding collection is an error rather than a hidden rollout.
Successor targets contain occupancy only, so a reward update between cycles
never invalidates the critic — pessimism and the reward both enter at
scalarization.

One batch is `32 x 250 = 8,000` interactions. At UTD 2 that is 16,000 critic
optimizer steps, split evenly across 16 policy improvements, each of which
fits one frozen projection target with up to 8 actor steps. The 1,000-step
initial critic warm-up runs **before the first online collection**, samples only
the protected demonstrations, and updates both successor and entropy critics
against the frozen BC policy. Feature scales are initialized from those demos.
The actor does not change, no online UTD credit is earned, and `model_0.pt` saves
this initialization. Once online data arrives, replay uses the configured 50/50
demo/online mixture. Without demos, the existing replay warm-up timing is unchanged.
Warm-up is computation, not interaction, and is logged under `WarmStart/*`.

### Reading the feature gap

The measured `mu(pi)` belongs to the policy that **collected** the batch, not
to the policy the update block leaves behind. `IRL/collection_policy_version`
is logged next to it for exactly that reason.

This matters when reading the reward-step ablation. Raising
`irl.gradient_steps` from 8 to 64 fits `w` more exactly to a *fixed* measured
gap; it does not make that gap smaller within the cycle. A run whose feature
gap does not shrink across cycles is telling you about the policy, not about
the reward optimizer.

## Data and compute accounting

Per online run:

| | |
| --- | --- |
| Physical training interactions | 100 cycles x 8,000 = **800,000** |
| Critic optimizer steps | 100 x 16,000 = 1,600,000, plus 1,000 warm-up |
| Policy improvements | 100 x 16 = 1,600, each up to 8 actor steps |
| Reward optimizer steps | 100 x 8 = 800 (or 6,400 in the follow-up) |
| Demonstrations | 10 episodes = 2,500 transitions, shared by every arm |

Recording costs more than it keeps: attempts that did not become
demonstrations are still simulator interactions, and `RecordingCost` reports
them split into the two kinds that mean opposite things - attempts that
*failed*, and attempts that succeeded past the quota. Ten demonstrations from
one 64-lane wave cost 16,000 interactions and keep 2,500. Evaluation is counted
separately (`Summary/evaluation_transitions`) and is never training data.

A high update-to-data ratio is a compute choice, not evidence of sample
efficiency. The comparison axis is physical training interactions.

## Running it

**1. Qualify the expert.** The recorder *is* the qualification instrument: it
accepts exactly the episodes that meet the task's own rule, so one wave of 64
attempts reports the success rate directly. The saved-demo quota does not
truncate the success count: failed attempts and surplus successes are reported
separately. Bound qualification to one complete wave:

```bash
python scripts/record_demos.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --no-video --num_envs 64 --num_demos 1 --max_attempted_episodes 64 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --teacher_config <teacher>.yaml --checkpoint <teacher>.pt \
  --output_dir /tmp/qualify
```

The bar is 95% of 64 complete attempts. The candidate teacher
(`lift_easy_entropy_quantiles_20260919_BRLXuf/.../model_1500.pt`) measured
**64/64** on 2026-09-22 with executed actions bounded at +/-4 (largest recorded
|action| 3.11). If a teacher does not qualify, report that rather than changing
the expert or the task.

**2. Record ten demonstrations.** The recorder reads the *arm's* config, so the
demonstrations are recorded on the task they will be used for, and it keeps only
complete successful episodes:

```bash
python scripts/record_demos.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --teacher_config configs/lift_cube_franka/ppo_easy.yaml \
  --checkpoint <ppo-teacher>.pt --num_demos 10 --num_envs 32 \
  --output_dir logs/demos/lift_sparse
```

`--teacher_config` exists because the teacher was trained without the episode
clock; the recorder cuts the observation back to the width that teacher saw.
Recording refuses to overwrite an existing file.

**3. Fit the shared actor**, once per seed, with no simulator:

```bash
python scripts/validate_bc.py --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --set seed=42 --expert_data_path logs/demos/lift_sparse/demos.hdf5 \
  --gradient_steps 40000 --learning_rate 3e-4 --final_learning_rate 3e-6 \
  --l2_candidates 0 1e-6 1e-5 --folds 5 --batch_size 256 \
  --output logs/bc/lift_sparse_s42.pt
```

This checkpoint *is* the BC arm, and it is the initialization the other three
arms are evaluated against at zero online interactions.

The original 10k-step fixed-learning-rate fit did not qualify in closed loop. The subsequent
[BC calibration](bc_calibration.md) found a seed-42 deterministic baseline at
30/64 successes using 20k steps at 3e-4 plus 1k refinement steps at 3e-5.
That success-band selection rule was subsequently rejected in favor of fitting
BC fully with explicit regularization and whole-trajectory validation, as in
the command above and [the BC fitting protocol](bc_regularization.md). Do not
use the 47% checkpoint as the benchmark merely to leave room for RL. Stochastic
collection and seeds 43/44 are not yet qualified.

The replacement seed-42 full fit is
`logs/benchmarks/lift_bc_regularized_20260922/bc_s42.pt`: 52/64 deterministic
successes on fresh reset seeds 5001/5002. Five-fold trajectory validation chose
zero L2 over 1e-6/1e-5; the actor completed 40k updates with cosine learning-rate
decay on all ten demos. See the fitting protocol for exact scores and costs.

**4. Run an online arm.** Both demonstration paths name the same file; a
mismatch is refused at config load:

```bash
DEMOS=logs/demos/lift_sparse/demos.hdf5
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --seed 42 --expert_data_path $DEMOS \
  --set algo.pmd.train.replay.demonstration_path=$DEMOS \
  --init_policy_checkpoint logs/bc/lift_sparse_s42.pt
```

Seeds 42, 43, 44. At most two learners on the GPU at a time. The
64-reward-step follow-up is the same command plus
`--set irl.gradient_steps=64`.

**5. Evaluate externally**, on the saved checkpoints, after the learners have
released the GPU. In-run validation is off, so evaluation never resets the
training environment. Checkpoints land every 5 cycles, which covers 80k
(cycle 10), 200k (25), 400k (50) and 800k (100) interactions. Score 64
deterministic plus 64 stochastic episodes per checkpoint with reset seeds 1001
and 1002. Standard-reset transfer is a final secondary evaluation, never a
selection criterion.

For finite-horizon successor PMD, the standalone evaluator uses the shared task
builder, pre-reset tap and complete-episode collector:

```bash
python scripts/evaluate_checkpoints.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --expert_data_path logs/demos/lift_sparse_qualify/demos.hdf5 \
  --checkpoints logs/train/<run>/model_0.pt logs/train/<run>/model_10.pt \
  --output_dir logs/evaluations/<new-directory> \
  --num_envs 32 --episodes_per_seed 32 --evaluation_seeds 1001 1002 --logger wandb
```

Use the original arm's config and any task/network overrides from its launch.
The output directory must be new. Evaluation loads actor, successor, entropy
and learned-reward weights, but performs no warm-up, replay insertion or
optimizer steps. Deterministic and stochastic performance are separate. Each
checkpoint receives 64 episodes of each mode; all vector-lane interactions,
including discarded transitions after failures, are counted as evaluation cost.
W&B publishes a separate evaluation run with both modes on the explicit
`Samples/training_environment_interactions` axis, including the zero-interaction
BC baseline. JSON results preserve per-seed counts and checkpoint paths.

Only **stochastic** episodes diagnose the stochastic-policy critics. The evaluator
compares raw successor predictions with valid feature suffix sums, and the
entropy critic with the sum of `-dt * log pi` **after** the current action.
It reports reward, sparse, learned and entropy components, per-head means,
head disagreement, bias and sampled-return RMSE, at episode starts and across
valid transitions. These errors include Monte Carlo return variance; they are
not exact conditional-Q errors or a proof of convergence. The final critic can
also lag the last actor projection. Terminal feature errors have no future
return noise or bootstrap. Continuous-action differential entropy may be negative.

Mixed replay batches are shuffled jointly across every observation, action,
feature and boundary field before contiguous head slicing. Thus every head
sees the same demonstration/online mixture in expectation, with an exact
demonstration quota for the whole batch. Training and episode validation use
one success reader: the terminal pulse on the final valid transition, with
failure taking precedence over timeout; continuing tasks retain ever-reached
`success_bonus` semantics.

## Reporting

One W&B comparison on actual physical training interactions, with BC as the
zero-online-interaction reference. Report the training trajectories, the
discarded vector lanes, the demonstration and evaluation costs, the actual
update counts, the sparse/learned/entropy reward components separately, and the
collection-policy feature gaps.
