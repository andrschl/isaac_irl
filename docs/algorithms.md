# Algorithms

A run can combine reward learning, policy learning, and an imitation regularizer.
This document is the overview; [PMD policy and value updates](pmd_updates.md)
is the equation reference for the off-policy learner.
For configuration and validation of a new task, see
[Adding an environment](adding_environments.md).

- **The reward** (IRL): a model fit to the gap between expert and imitator
  feature returns.
- **The policy** (RL): `pmd`, an exponential policy-mirror-descent target
  followed by a bounded neural actor projection, or `ppo`, the on-policy
  rsl-rl baseline.
- **The regularizer** (BC): an imitation loss mixed into the policy update.

PMD accepts either of two value representations - a scalar direct Q or a
reward-independent successor-feature Q - over the same bounded tanh-Gaussian
actor, transition replay, and data collector, so a comparison between them
changes one thing.

Observations are consumed as the environment reports them. A running
standardizer was removed after it caused a Lift failure worth recording: its
statistics accumulate over the whole run, so by a million transitions a new
rollout moves them by 0.1% and the transform is frozen on the distribution the
policy has already left. On Lift that froze the object-height scale at a
standard deviation of 0.01 m — collected while the cube never left the table —
so lifting the cube 10 cm landed exactly on the +-10 clip, and the one
observation that reports task progress saturated at the moment it started to
matter. Isaac's observation terms are already O(1); the tuned on-policy
baseline runs on them unnormalized.

## PMD

PMD evaluates a frozen policy from transition replay, scores candidate actions,
and fits a tanh-Gaussian actor to an exponential improvement target. Candidate
weights and the fitted actor have separate KL constraints. The critic targets
use Polyak averaging; the reference policy is replaced after each projection.

Choose the value representation with `algo.value.type`:

- `direct` learns scalar action values and rescores replay features with the
  current reward. Its default TD backup clips a random target-head subset;
  control takes the minimum over all online heads.
- `successor` learns reward-independent feature returns. A linear reward
  scalarizes each head before a lower-quantile control reduction. Changing
  reward weights changes scores without changing the vector TD targets.

The [PMD reference](pmd_updates.md) specifies the losses, reductions, optional
entropy term, projection constraints, update order, and experiment defaults.
Configuration fields and validation live in [schema.py](../src/config/schema.py).
For Lift-specific architecture choices, action scaling, the comparison with
Raffin's SAC article, and what remains unvalidated, see
[action bounds and architecture rationale](lift_action_bounds.md).

## Shared entropy-regularized objective

Both policy algorithms interpret `entropy_rate` as the same $\alpha$ in
$\sum_t\gamma^t\Delta t[r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))]$.
PPO adds the sampled $-\Delta t\alpha\log\pi(a_t\mid s_t)$ to rollout rewards
before GAE and value-target construction; the rsl-rl optimizer entropy term is
set to zero to avoid counting it twice. Successor-PMD keeps task successor
features independent and learns a separate $H(s,a)$ for the integrated future
entropy return. Its control value is $Q_R+\alpha H$, while policy improvement
adds the current candidate's $-\Delta t\alpha\log\pi_k$ term. Direct PMD puts
the same term directly in its scalar soft-Q target.

## Transitions, terminals, and time limits

Replay stores every transition with its own successor observation rather than
shifting one observation sequence. Isaac computes rewards before it resets the
environments that finished, so the [pre-reset tap](../src/adapters/pre_reset_tap.py)
captures terminal observations. The [PMD reference](pmd_updates.md#1-scope-and-notation)
defines which transitions bootstrap. The extra observation pass is validated
once per run: environments that did not reset must observe the same state
inside and after the reward manager, or the run stops with an explicit error.

## What a run reports

Every run emits the same `Summary/*` contract regardless of algorithm or reward
source, so two runs are read the same way. The x-axis is
`Summary/environment_transitions`—all environment interactions, including
evaluation and unused vector lanes. `Summary/transitions` separately reports
retained training data. See [episode training](episode_training.md) for budgets
and the native PPO benchmark (which disables inline validation).

Task performance comes from a deterministic validation rollout, scored on the
environment's own reward whatever the policy was trained against; the sampled
policy is scored too, because a deterministic collapse and a policy collapse
look identical otherwise. `Summary/feature_gap_norm` is
`||E_expert[phi] - E_imitator[phi]||`, the quantity IRL is minimizing, and
`Summary/reward_weight_cosine` compares the learned linear reward against the
environment's own weights when the task exposes them.

Detailed diagnostics keep their own namespaces and are not summarized:
`PMD/*` (projection KLs, candidate statistics), `DirectQ/*` / `SuccessorQ/*` /
`EntropyQ/*` (per-head TD losses, scales), `Data/*` (replay occupancy, policy
age, termination fractions), `Action/*` (saturation), `Reward/*` (per-term
contributions), `ValidationFeatures/*` (per-feature expert/imitator/gap).

## IRL

The reward model is fit to

`L_reward = E_imitator[R] - E_expert[R] + regularization`,

where each return is a discounted sum of per-step features scored by the model,
optionally normalized by episode length. Features are the environment's own
unweighted reward terms, captured inside the reward manager before Isaac's
auto-reset so a terminal step is recorded at all.

PMD requires `runner.imitator_source=collected`: its collection already returns
complete episodes and a padding mask, so IRL reuses those exact attempts without
extra interaction. Failures are included, not filtered by success or length.
PPO still collects short rollouts; its `collected` source contains fragments,
and its optional `rollout` source pays for a dedicated rollout. That PPO/IRL
variant is distinct from the uninterrupted native environment-reward benchmark.

## Behavioral cloning

The config name is `bc.weight` (CLI: `--bc_weight`). BC supplies an expert
negative log-likelihood loss (`bc.loss_type: nll`) or a mean squared error on
deterministic actions (`mse`), implemented by
[BC.compute_policy_loss](../src/algorithms/bc.py).

`L_actor = (1 - bc.weight) L_RL + bc.weight L_BC`, with `bc.weight = 0`
constructing no loader, batch, or gradient, and `bc.weight = 1` training the
policy on demonstrations alone without stepping the environment. The validation
loss is computed over the whole held-out split, not a sampled batch of it.
