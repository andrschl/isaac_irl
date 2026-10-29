# Episode-batch PMD and the PPO benchmark

For the opt-in pre-grasp reset and nearby-goal task, see [easier Lift](lift_easy.md).

PMD is organized around complete attempts at the task. PPO keeps its standard
short-rollout update. These are different collection protocols, not two settings
of one rollout-length parameter.

## One PMD iteration

1. Freeze the behavior policy and collect `train.episodes_per_batch` complete
   episodes. A genuine failure counts as an episode, just like a timeout.
2. Put valid transitions in replay. If learning a reward, reuse the same
   `features [B,T,D]` and `mask [B,T]` for the IRL update. There is no additional
   imitator rollout. Reward weights then stay fixed during the policy update.
3. Resolve the critic update budget, then divide it into interleaved critic-fit
   and bounded policy-improvement rounds. Each round evaluates the policy
   produced by the preceding round; the first evaluates the collection policy.
4. Log/checkpoint, and evaluate if due. Evaluation finishes before the next
   training episode batch begins.

Here `B` is the episode count, `T` the environment's maximum episode length,
and `D` the feature dimension. Padding is masked out of replay, returns, success
rates, and IRL. The default feature buffers sample episodes uniformly rather
than weighting an attempt by its duration; one-step failures are retained.

The actor does not change during collection. During learning, its frozen
previous-policy snapshot is the critic's policy-evaluation target and is
refreshed after every improvement. Full-episode collection does not restrict
how many improvements can use that replay. A projection can take several actor
optimizer steps on one fixed target; those are not separate improvements.
See [the exact equations](pmd_updates.md).

### Independent data and compute budgets

Current Lift settings, relative to `algo.pmd`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `train.episodes_per_batch` | 32 | Completed attempts per collection |
| `train.critic_updates_per_batch` | 1000 | Regular critic optimizer steps per batch |
| `train.critic_updates_per_transition` | null | Alternative: critic steps per newly retained transition |
| `policy_improvements_per_batch` | 1 | Bounded policy projections per batch |
| `projection_steps` | 8 | Maximum actor optimizer steps within each projection |
| `train.initial_critic_updates` | 1000 | Extra critic-only warm-up, once |
| `train.replay.learning_starts` | 4096 | Minimum retained transitions before learning |

For `C` critic steps and `P` improvements, require `1 <= P <= C`. Split `C`
as evenly as possible across `P` evaluate-then-project groups: the first
`C % P` groups get one extra critic step. All `C` steps execute; none are lost
to rounding. Warm-up runs before the regular schedule in the first eligible
batch, with the actor unchanged during warm-up.

Choose exactly one critic budget; set the other to `null`. A fixed per-batch
budget does not grow with episode length. A transition-based budget (UTD)
scales with retained data, not vector steps, ensemble heads, discarded-lane
interactions, evaluation, or the existing replay size. Fractional update credit
carries between eligible batches and is checkpointed. Batches collected before
`learning_starts` accrue no update debt; the one-time initial fit is additional.
The policy budget must fit even if every episode ends after one step.

Changing parallelism alone does not change these budgets for the same retained
episodes. The defaults are conservative starting points, not a validated claim
that PMD learns Lift.

### High-reuse scheduling experiment

The opt-in [high-reuse Lift config](../configs/lift_cube_franka/pmd_high_reuse.yaml)
uses fixed environment rewards, 32 episodes, UTD 2, and 32 improvements. A full
32-by-250 batch produces 16,000 regular critic steps: 32 groups of 500 critic
steps plus one projection, with up to eight actor optimizer steps per projection.
The 1,000-step initial critic fit remains additional. Short episodes reduce the
critic budget proportionally; they do not discard old replay.

This changes scheduling only relative to the fixed-reward PMD baseline. It
keeps the width-512, two-residual-block, three-head successor critic, Adam,
target updates, action distribution, and `control_quantile: 0.25`. It does not
add weight decay, larger critics, or parameter resets. A quantile above 0.5
(1.0 selects the maximum head) is optimistic scoring, but it is not BRO's
separate optimistic exploration policy; keep it fixed when isolating scheduling.

Monitor `Data/critic_updates_per_transition` (regular UTD),
`Data/critic_updates_per_environment_transition` (including unused training
lanes in the denominator), and `Data/actor_optimizer_steps`. Initial critic
fitting remains separately logged. `PMD/collection_policy_kl` is the final
KL(collection policy || updated policy), averaged over a fixed, evenly spaced
subset of retained states. Its mean/spread components are logged separately.
This is a drift diagnostic, not a batch-wide constraint: the existing KL bounds
apply to each individual improvement and do not bound cumulative drift.

BRO's reference settings use UTD 10 (2 for BRO-Fast) and update both actors in
every critic round. That motivates testing higher reuse, not assuming these
settings reproduce BRO: it also uses strong critic regularization and a
different exploration mechanism. [BRO defaults](https://github.com/naumix/BiggerRegularizedOptimistic),
[update loop](https://github.com/naumix/BiggerRegularizedOptimistic/blob/main/jaxrl/bro/bro_learner.py).

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/pmd_high_reuse.yaml
```

This config disables inline validation so separate-process checkpoint evaluation
does not interrupt training. Use the same held-out seeds and success definition
as PPO, and account for external evaluation interactions separately.

The old `train.rollout_steps`, `train.updates_per_transition`, and
`critic_updates_per_projection` keys are rejected, not silently translated.
`runner.steps_per_env_per_cycle` now concerns PPO only. PMD requires
`runner.imitator_source=collected` and `runner.rl_updates_per_cycle=1`.

## Complete episodes in one Isaac environment

We do not create a second simulator or attempt to restore an incomplete
physics snapshot. The [episode collector](../src/adapters/episode_collector.py)
collects waves of at most `N = env.num_envs` episodes:

- Reset explicitly at the start of a wave.
- Retain the first episode from each selected lane, through its first ending.
- Wait for all selected lanes to finish before starting another wave or
  returning the batch. Never stitch across a reset.

Isaac still steps every lane. If one lane finishes early, its subsequent
auto-reset transitions are excluded from training but counted as interactions.
A partial final wave also pays for unused lanes. This avoids preferentially
selecting only the fastest/shortest episodes. Use `N <= B`, ideally with `B`
divisible by `N`, to reduce overhead. A single robot (`N=1`) has no unused-lane
overhead; robot reset/settling time remains a separate physical cost.

This change does **not** disable safety terminations or change TD targets.
True terminations stop bootstrapping. Time-limit truncations still bootstrap
from the pre-reset successor observation. Collecting whole episodes does not
by itself turn the critic into a finite-horizon value function.

## Validation without interrupting a training episode

For PMD, each validation collects `runner.validation_episodes` complete episodes
per mode (deterministic and stochastic). It runs only after the collected
training batch has been consumed by learning. It never enters replay or the
imitator buffer. No episode-age staggering is used for PMD.

The runner invalidates cached observations after evaluation but does not issue
an extra reset. The next training wave owns its reset. This removes the old
mid-episode validation interruption and hidden restart path.

Global PyTorch, NumPy, and Python RNG states are restored around each PMD
evaluation; the private training-action generator is not consumed. This is not
a promise of bit-identical simulator trajectories with/without validation:
arbitrary environment-private state/RNGs and simulator counters are not restored.

`validation_interval` counts completed training batches. Alternatively,
`validation_interval_transitions` triggers after a retained-training-transition
threshold is crossed, still at a batch boundary. Set the interval to zero and
the transition interval to null to disable inline evaluation. For PMD,
`validation_steps_per_env` does not shorten episodes (zero disables evaluation).

## Count the interactions actually paid for

| Metric/config | Meaning |
| --- | --- |
| `Summary/transitions` | Retained training transitions |
| `Summary/environment_transitions` | All training, evaluation, dedicated imitator, and unused-lane interactions |
| `Summary/evaluation_transitions` | Evaluation interactions, including final video rollouts |
| `Summary/training_episodes`, `Summary/evaluation_episodes` | Completed selected PMD episodes; PPO training counts observed endings (initial ages may be staggered) |
| `Data/discarded_transitions` | Collection interactions excluded from replay in this batch |
| `Time/collection_seconds`, `Time/reset_seconds` | Collection wall time (including resets), and its reset portion |
| `Time/aggregate_interaction_seconds` | All collection interactions times the environment control timestep; not wall time |
| `runner.total_transition_budget` | Hard limit on all counted environment interactions |
| `runner.transition_budget` | Legacy retained-training-transition stopping threshold; may overshoot by a batch |

W&B's default `Summary/*` x-axis, AUC, and time-to-threshold metrics use total
environment interactions. Training-only counts remain available separately.
Reset-internal physics/settling steps are not fabricated as agent transitions;
the reset wall-time metric makes part of that additional cost visible.

Before starting collection/evaluation, reserve its worst-case cost: for `B`
episodes, `ceil(B/N) * N * T` interactions. Evaluation reserves both modes.
If the next batch cannot fit, stop training or skip evaluation rather than
truncate an episode. Consequently the run can stop below its budget, especially
when failures shorten episodes. Final video rollouts must fit too. Counts are
checkpointed; old checkpoints without the ledger cannot resume a claimed hard
budget (actor-only initialization remains available).

For Lift, 32 episodes of 250 steps cost 8,000 training interactions if none
end early. Ten validation episodes per mode with 32 lanes still cost 16,000
interactions across both modes. Increase the selected evaluation count to 32
to use all those lanes, reduce parallelism, or evaluate less often. Evaluation
is not free data just because it is excluded from replay.

## Preserve standard PPO, and make curriculum explicit

[Lift's native PPO config](../configs/lift_cube_franka/ppo.yaml) keeps the
task's 4,096 environments, 24-step rollouts, unbounded Gaussian actor, PPO
hyperparameters, and reward-weight curriculum. Inline validation is disabled so
it does not disturb PPO's training episodes. Use this config explicitly:

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/ppo.yaml

# Same PPO implementation with fixed reward weights:
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/ppo.yaml \
  --set env.reward_curriculum=false

# PMD: whole episodes, fixed reward, explicit total interaction budget:
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --reward_source environment --num_envs 32 \
  --set algo.pmd.train.episodes_per_batch=32 \
  --set algo.pmd.train.critic_updates_per_batch=1000 \
  --set algo.pmd.policy_improvements_per_batch=1 \
  --set runner.total_transition_budget=200000
```

Why was curriculum disabled? Lift changes the action-rate and joint-velocity
reward weights from `-1e-4` to `-0.1` after 10,000 vector control steps (the
curriculum is applied at resets). Different parallelism means that switch costs
different numbers of transitions. More importantly, PMD's fixed linear reward
and the pre-reset feature tap assume fixed reward weights. A changing reward
definition needs explicit handling in replay scoring and feature extraction;
turning off those checks would not implement that handling.

`env.reward_curriculum=true` is therefore supported for environment-reward PPO
without diagnostic reward fitting. PMD/IRL reject it with an actionable error.
False removes known `modify_reward_weight` curriculum terms, not unrelated
task curricula. `RewardWeights/*` logs PPO's active term weights. Demonstration
recording still freezes weights to keep recorded feature semantics stationary.

Native PPO is a task-performance reference, not automatically a matched
sample-budget comparison: one 4,096-by-24 rollout already costs 98,304
transitions, and curriculum changes its objective. Report both native PPO and
a separately labeled fixed-reward/sample-budget PPO variant if needed. Use
common held-out task-success evaluation; do not equate raw returns under
different reward weights. The standard PPO benchmark itself is not replaced
by an episode-batched PPO variant.

## Verification

Unit tests cover variable-length/one-step episodes, wave overhead, timeout
bootstrapping, IRL mask reuse, independent update budgets, boundary-only
evaluation, RNG isolation, hard-budget reservation, and curriculum configuration.
Real-Isaac smoke tests exercise both value backends and PPO. A wiring smoke
test checks finite updates, episode counts, and saved checkpoints—not whether
the policy has learned to lift.
