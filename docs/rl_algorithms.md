# RL algorithms

The training API exposes exactly three policy algorithms:

- `ppo`: the standard on-policy rsl-rl implementation.
- `sac`: a reparameterized, maximum-entropy off-policy update.
- `pmd`: an exponential policy-mirror-descent target followed by a neural
  actor projection.

SAC and PMD share the same bounded tanh-Gaussian actor, transition replay,
observation normalization, data collector, and selectable value backend. This
keeps sample-efficiency comparisons controlled.

## Actor losses and BC

Behavioral cloning is an independent regularizer for all policy updates:

`L_actor = (1 - bc_weight) L_RL + bc_weight L_BC`.

The config name is `bc.weight` (CLI: `--bc_weight`). At its default value
of zero, no expert observation/action loader, BC batch, or BC gradient is
constructed. It can be enabled later without changing the RL algorithm.

## SAC

For a reparameterized action `a = f_theta(s, epsilon)`, SAC minimizes:

`L_SAC = E[entropy_temperature * log pi(a|s) - Q_soft(s,a)]`.

`initial_entropy_temperature` is the entropy coefficient. With
`automatic_entropy_tuning=true`, it is the initial value and is subsequently
learned. SAC has no old-policy snapshot and no actor KL constraint.

The unconstrained density target is the `beta=0` member of the exponential
family below. The implemented SAC optimizer remains reparameterized rather
than approximating that target with finite candidates.

## PMD

PMD freezes `pi_old` for one critic/projection cycle and defines:

`pi_target(a|s) proportional to pi_old(a|s)^beta * exp(c * Q(s,a))`.

Candidates are sampled from `pi_old`, so weighted maximum likelihood uses
the proposal-corrected log weight:

`log u = (beta - 1) * log pi_old(a|s) + c * Q(s,a)`.

One update is one *cycle*: sample a replay batch, draw `candidate_actions`
actions per state from `pi_old`, solve `c`, then take up to
`projection_steps` weighted-NLL steps against that fixed candidate target,
stopping as soon as the actor reaches `actor_projection_kl`. Candidate actions
keep their pre-squash latents, so the projection never inverts `tanh`.

Sample efficiency is decided by how often that cycle runs:

- `critic_updates_per_projection`: critic steps that evaluate the frozen
  `pi_k` before it is replaced. One rollout of `rollout_steps * num_envs`
  transitions yields
  `critic_updates_for(transitions) / critic_updates_per_projection`
  projections, each worth at most `actor_projection_kl` nats of policy
  movement. Lowering it buys policy improvement per simulator transition with
  compute, exactly as `updates_per_transition` does for the critic.

Important parameters:

- `beta`: old-policy exponent. `1` is the usual mirror-descent prior;
  `0` is the SAC/Boltzmann density target.
- `candidate_actions`: candidate count per replay state.
- `candidate_target_kl`: KL bound for the finite nonparametric target. A
  bisection chooses the largest admissible Q coefficient `c`.
- `actor_projection_kl`: hard trust region on the fitted actor's KL from
  `pi_old`. A single optimizer step can overshoot it by orders of magnitude,
  so the step that crosses the bound is shrunk along its own displacement by
  up to `projection_backtrack_steps` halvings until the bound holds.
- `actor_spread_projection_kl`: the same bound applied to the part of the KL
  that only widens or narrows the action distribution. One bound on the whole
  Gaussian cannot separate "moved the actions" from "made them noisier", and a
  weighted fit to its own samples drifts outward: on Lift the unconstrained
  version reached a latent standard deviation of 5 — saturated, bang-bang
  actions — while its mean barely moved. `PMD/actor_mean_kl` and
  `PMD/actor_spread_kl` report the two parts.
- `projection_steps`: maximum weighted-NLL optimizer steps per projection.
- `actor_learning_rate`, `actor_batch_size`: projection optimizer step size
  and the number of replay states carrying the candidate target.
- `q_coefficient_min/max`: numerical search bounds for `c`.

Q values are never z-normalized. Additive shifts cancel in normalized weights,
while multiplicative reward changes deliberately change the update.

## Value backends

SAC and PMD accept `algo.value.type: direct | successor`.

### Direct Q

The direct backend learns scalar soft action values:

`target = reward + gamma * mask * (target_Q - entropy_temperature * log pi)`.

Every ensemble head learns from a disjoint slice of one replay draw and
bootstraps from its paired EMA target. Control uses the ensemble mean; there is
no clipped minimum or pessimistic reduction. Drawing once and splitting keeps
the per-head data independent while costing one sample and one next-action pass
per update instead of one per head.

### Successor Q

Each head predicts a vector in raw feature-return units:

`Psi_j(s,a) = E_pi[sum_t gamma^t phi_t | s_0=s, a_0=a]`,

with reward-independent target:

`target_Psi_j = phi + gamma * mask * target_Psi_j(s',a')`, where
`a'` is sampled from the target policy.

A linear reward is applied only at action scoring time:

`Q_reward(s,a) = step_dt * dot(w, mean_j Psi_j(s,a))`.

Changing `w` therefore changes action rankings immediately and never changes
successor TD targets.

When entropy is enabled, a separate scalar return critic learns:

`target_H = gamma * mask * (-log pi(a'|s') + target_Q_H(s',a'))`,

and `Q_soft = Q_reward + entropy_temperature * Q_H`. IRL feature occupancy
uses only `Psi`, never `Q_H`.

## Shared off-policy parameters

These live under `algo.train`:

- `gamma`: Bellman discount.
- `rollout_steps`: simulator steps collected per environment per cycle.
- `updates_per_transition`: update-to-data ratio (UTD). `1/16` means one
  critic update per 16 new simulator transitions.
- `initial_critic_updates`: critic-only warm-up after
  `replay.learning_starts` has been reached.
- `critic_batch_size`, `critic_learning_rate`: TD optimizer batch and step.
- `target_polyak`: EMA interpolation coefficient; `0.005` means targets
  move 0.5% toward online parameters after each critic step.
- `replay.learning_starts`: minimum stored transitions before any learning.
- `replay.capacity`: maximum stored transitions; replay is a ring buffer,
  so the oldest transitions are overwritten once it is full.
- `replay.storage_device`: `auto` keeps the ring on the training device, which
  removes a host-to-device copy from every sample; `cpu` trades that back for
  host memory. A ring that would not fit is refused with the measured bytes
  per transition.
- `policy.maximum_std`: ceiling on the actor's latent spread. A weighted fit to
  a policy's own samples inflates variance while the critic is uninformative,
  and a per-step trust region cannot stop the drift compounding: on Lift the
  unbounded policy reached a spread of 4.7 and flailed.
- `observation_epsilon`, `observation_clip`: running-normalization
  numerical floor and normalized observation bound.

All sample-efficiency reports should use simulator transitions on the x-axis.
Increasing UTD changes compute per transition, not the transition count.

## Transitions, terminals, and time limits

Replay stores every transition with its own successor observation rather than
shifting one observation sequence. Isaac computes rewards before it resets the
environments that finished, so the pre-reset tap
(`adapters/rsl_rl_pre_reset_tap.py`) captures the true terminal observation of a
time-limited episode; the following transition still starts from the post-reset
state. Consequently:

- a genuine termination stops the backup (`bootstrap_valid` is false),
- a time limit bootstraps from the real terminal state, matching what rsl-rl
  PPO does with `extras["time_outs"]`,
- the extra observation pass is validated once per run: environments that did
  not reset must observe the same state inside and after the reward manager,
  otherwise the observation pipeline is stateful and the run stops with an
  explicit error.

## W&B overview

Fixed-reward runs emit a deliberately small `Overview/*` contract in addition
to detailed algorithm diagnostics. The primary performance metric is
`Overview/return_per_step`, which makes evaluation horizons directly
comparable; tasks with a success metric add `Overview/success_rate` and
`Overview/transitions_to_success_rate_*`. Every task uses deterministic
250-step validation by default, and its profile
(`src/config/benchmark.py`) fixes the environment count, transition budget,
thresholds, and the tuned PPO baseline it is compared against.

Create the curated report with:

```bash
python scripts/rl/create_wandb_overview.py --group lift-fixed-reward-v4
```

The report groups seeds by the explicit `variant` config value: `ppo`,
`sac-direct`, `sac-successor`, `pmd-direct`, or `pmd-successor`. Detailed
metrics remain available in W&B but are kept outside the main view.

`--entity` defaults to `$WANDB_ENTITY`.
