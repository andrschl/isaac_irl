# Franka Lift: action bounds and architecture rationale (2026-09-19)

## Decision

Set only `algo.pmd.policy.action_bound` in the Lift experiment to **4.0**.
The existing actor already implements the correctly normalized density for
`action = bound * tanh(latent)`. No new clipping layer or environment patch is
needed. Cartpole, PPO, critic targets, ensemble reduction, reward weights,
initial latent standard deviation, and update cadence are unchanged.

Isaac's `JointPositionActionCfg(scale=0.5, use_default_offset=True)` means

```
arm_joint_target = default_joint_position + 0.5 * arm_action
```

The target is **not** an increment from the current position. The previous
bound of 1 allowed targets only +/-0.5 radians from the default pose; 4 allows
+/-2 radians. These are commanded targets, not guarantees about actual joint
positions under gravity, contacts, and joint limits. PPO has no corresponding
action clip in our wrapper. The gripper still opens/closes by sign; its raw
action magnitude remains visible in observations and action-rate penalties.

## Relationship to Raffin's SAC article

**Consistent with its action-space design principle, but not a reproduction
of its SAC configuration or a validated PMD learning solution.**

Raffin's Unitree example rescales a squashed Gaussian to +/-100, producing
much broader initial commands than PPO. Reducing that range, using a
task-PPO-like network, and reducing initial entropy pressure enabled learning.
The shown network is `[512, 256, 128]` with ELU; its example uses
`ent_coef="auto_0.006"` and UTD 0.25. The takeaway is to examine actual action
distributions and task-appropriate bounds, not to copy a universal bound.
[Source: Part I, action distributions and quick fix](https://araffin.github.io/post/sac-massive-sim/).

Our diagnosed mismatch goes in the other direction: +/-1 severely restricted
the working Lift expert. Expanding it to +/-4 follows the same principle of
checking the task's effective control range. This comparison concerns the
linked Part I, not an implementation of all suggestions from Part II.

### What our architecture actually is

The following describes the current resolved Lift config and implementation,
not settings established as optimal by the article. Config keys are relative
to `algo.pmd` unless otherwise stated.

| Component | Current choice | Assessment |
| --- | --- | --- |
| Actor hidden layers | `[256, 128, 64]`, ELU | Matches our **Lift PPO** baseline's hidden architecture, not the Unitree network widths. |
| Actor output | Separate state-dependent mean and log-standard-deviation heads; scaled tanh | Shares a hidden architecture with PPO, not its action distribution. Our PPO has an unsquashed Gaussian with state-independent learned standard deviation. |
| Action range | `policy.action_bound: 4.0` in native task units | Explicitly selected and tested; not inferred from a permissive simulator action-space box. |
| Initial exploration | `policy.initial_std: 1.0`, latent std limited to `[0.05, 2.0]` | Unchanged by the bound fix; appropriateness at the wider range is **not yet established**. |
| Entropy objective | `entropy_rate: 0.0` | No explicit entropy bonus or learned SAC temperature; does not make collection deterministic. |
| Critic | Three successor-feature heads on a shared width-512 residual trunk with two residual blocks, LayerNorm and ReLU; latent width 128; head hidden width 128 | A separate PMD/IRL design choice, not the article's plain MLP recipe. Shared features mean the heads are not independent networks. |
| Value reduction | Scalarize each successor head, then `control_quantile: 0.25` | Unchanged; neither this quantile nor critic ranking accuracy is validated by the action-bound experiment. |
| Observations | Native task observations, no running input standardizer | Matches our Lift PPO baseline's input treatment. Critic LayerNorm is a different operation from normalizing observations. |
| Data/update schedule | 32 envs, 32 complete episodes per collection, 1000 critic steps and one policy improvement per batch | See [episode training](episode_training.md); a later change from the short-chunk schedule used by the historical probes below. |

Implementation: [actor](../src/algorithms/off_policy/actor.py),
[PPO actor factory](../src/adapters/actor_critic.py),
[value networks](../src/algorithms/off_policy/values/networks.py),
[Lift config](../configs/lift_cube_franka/experiment.yaml).
Exact TD and projection equations remain in [PMD updates](pmd_updates.md).

### Why keep scaled tanh and use a configuration-only fix?

Let $s$ denote actor observations, $A$ the number of action coordinates, and
$\mu_\theta(s),\sigma_\theta(s)\in\mathbb R^A$ the actor's latent Gaussian
mean and standard deviation. With independent standard-normal noise
$\epsilon\sim\mathcal N(0,I_A)$, where $I_A$ is the $A$-dimensional identity,
we sample a latent action $z$ and execute native task action $a$:

$$
z=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,
\qquad a=b\tanh z,\qquad b=4.
$$

Here $\odot$ denotes coordinatewise multiplication and $b$ is the common
action bound. The actor, candidate scoring, replay, and environment all use
the same native action coordinates. There is no additional conversion of
these actions to the simulator's declared action-space bounds. Log densities
include both the tanh Jacobian and the scale term $A\log b$.

This retains the policy-density contract PMD uses for its weighted projection.
Adding hard clipping after sampling would change the executed distribution
without changing those density calculations. The expert probe clips only to
test feasibility; that clipping was **not** added to PMD training. The
[collector](../src/adapters/transition_collector.py) also rejects a wrapper
clip narrower than the actor's bound.

The hidden layers already match our task's PPO baseline. There was therefore
no reason to bundle a network rewrite with the action-range test. Changing
the action distribution, successor representation, or ensemble structure
would be separate algorithm experiments, not consequences of this fix.

### Action support and exploration are separate choices

The bound $b$ determines the available command range. The latent mean and
standard deviation determine where the policy samples inside it. For the
same latent action $z$, changing $b$ from 1 to 4 multiplies every executed
action by four. Near $z=0$, $a\approx bz$, so the local sensitivity and
small-noise action standard deviation also scale with $b$.

For illustration, consider **one zero-mean coordinate**, $z\sim\mathcal
N(0,\sigma^2)$. These values come from integrating the Gaussian marginal,
not from measuring our initialized network or simulator rollouts:

| Distribution | Native-action standard deviation | Central 95% action interval |
| --- | ---: | ---: |
| Previous bound: $a=\tanh z$, $\sigma=1$ | 0.628 | +/-0.961 |
| Current bound: $a=4\tanh z$, $\sigma=1$ | 2.512 | +/-3.844 |
| Illustrative smaller initial spread: $a=4\tanh z$, $\sigma=0.25$ | 0.945 | +/-1.817 |
| Reference unsquashed Gaussian, mean 0 and std 1 | 1.000 | +/-1.960 |

The values use $\operatorname{Std}(a)=b\sqrt{\mathbb E[\tanh^2 z]}$ for
the centered case; the central interval is approximately
$\pm b\tanh(1.96\sigma)$. The actual actor's initial mean is not forced to
zero, and its learned mean and standard deviation depend on the state.
The 0.25 example is a possible controlled ablation, **not a new default or an
exact match to PPO**. Preserving available actions does not require sampling
all of them aggressively at startup.

Zero entropy temperature removes an explicit incentive to maximize entropy;
it does not remove sampling noise. Our collection still samples the actor,
and the PMD projection can change both mean and variance. Its candidate,
actor, and spread KL budgets are `0.15`, `0.05`, and `0.002` respectively.
KL here measures a distribution change, not a physical displacement limit.
A common invertible action transform preserves KL, so an unchanged budget
does not compensate for wider native action scaling. Likewise, the latent
std floor does not prevent near-deterministic executed actions if the mean
drives tanh into saturation.

### Why this is not just a joint-reachability question

The article's appendix also notes that position targets outside joint limits
can be used to induce stronger PD-controller responses.
[Source: unbounded-action discussion](https://araffin.github.io/post/sac-massive-sim/#appendix---note-on-unbounded-action-spaces).
For our diagnostic, that means clipping can change both available target
poses and the forces used to approach them. It also changes last-action
observations. The experiment establishes disruption of a known controller,
not a proof that no controller can solve Lift under a smaller bound.

### What remains deliberately unvalidated

LayerNorm, a larger critic, and an ensemble do not establish accurate action
rankings. Likewise, matching PPO's hidden layers does not transfer PPO's
learning behavior to PMD. Under the historical short-chunk schedule, a post-warmup
collection provides $32\times8=256$ new transitions, followed by 512 critic
updates and **64 policy projections before the next collection**. This is a
substantial independent design choice; a successful expert action-range
check says nothing about its suitability. The current episode protocol instead
uses 32 complete attempts, 1000 critic steps, and one policy improvement per
batch. That change also needs learning experiments, not just integration tests.

Before claiming that this configuration learns reliably:

1. Measure initial and early-training native actions per arm joint, separately
   from the binary gripper. Check quantiles, velocities, and saturation, not
   only latent `policy_std`. Existing `Action/*` statistics pool coordinates;
   they are not already a per-joint exploration audit.
2. If exploration is too broad, test a smaller `initial_std` at the **same**
   bound, with fresh runs, matched seeds/budgets, and fixed environment reward.
   Do not change the critic or projection cadence in that comparison.
3. Independently check frozen-critic action rankings against sampled returns,
   then test projection cadence while holding the critic budget fixed.

These are proposed checks, not completed training results. The evidence
below covers the expert's action range and a short PMD integration run only.

## Measured evidence

Evaluate the existing deterministic PPO expert, without retraining, on
`Isaac-Lift-Cube-Franka-v0`. Checkpoint:
`logs/rsl_rl/franka_lift/2026-05-21_17-06-20/model_1499.pt`.
Its unnormalized 36-dimensional observation layout and native action mapping
match the installed task. Load its `actor_state_dict` MLP with ELU hidden
layers `[256, 128, 64]` and use the mean action.

For each seed (42, 43, 44), run eight conditions with 32 environments each.
Copy object pose/velocity and goal pose across conditions in local scene
coordinates; assert that their initial policy observations match. Clip the
expert's output before each environment step, so subsequent observations
include the actually executed action. Keep the normal environment dynamics,
disable reward-weight curriculum as training does, and run 249 steps to stop
before the 250-step auto-reset. Mask out dropped-object episodes after their
first termination. Ignore the first 25 steps for "ever" metrics to exclude
the initially falling cube.

All counts below are out of **96 episodes per condition**:

| Expert action constraint | Ever lifted above 10 cm | Lift held for final 25 steps | Ever within 8 cm of goal |
| --- | ---: | ---: | ---: |
| None | 96 | 96 | 78 |
| All coordinates +/-1 | 16 | 15 | 7 |
| Arm only +/-1; gripper unchanged | 28 | 28 | 16 |
| All coordinates +/-2 | 92 | 91 | 53 |
| Arm only +/-2; gripper unchanged | 86 | 85 | 59 |
| All coordinates +/-3 | 96 | 96 | 78 |
| All coordinates +/-4 | 96 | 96 | 77 |
| All coordinates +/-6 | 96 | 96 | 76 |

Goal success also requires cube height above 4 cm. At bound 4, mean episode
return was 127.51 versus 129.48 unrestricted (about 98.5%). Holding the goal
threshold throughout the last 25 steps was less robust: 62/96 versus 72/96.
Thus 4 is a tested starting range, not an optimized bound or full equivalence
to unrestricted control.

Bound 3 also retained every observed sustained lift. We chose 4 as an
intermediate range with higher mean return (127.51 versus 123.17 at 3) and
more sustained goal matches (62 versus 53). Bound 6 raised mean return only
to 128.59. This small experiment does not identify a statistically optimal
bound, a minimum sufficient bound, or optimal per-joint limits.

The initial hand-written IK controller was unsuccessful even with wider
bounds; it is not used as evidence for this decision. The trained expert
provides a working positive control instead.

Local run artifacts (under ignored `logs/`) preserve the exact probe and
per-seed data:
[probe.py](../logs/diagnostics/lift_action_bounds_2026-09-19/probe.py),
[results.json](../logs/diagnostics/lift_action_bounds_2026-09-19/results.json).
From the repository root, with the Isaac environment active, repeat with
`python logs/diagnostics/lift_action_bounds_2026-09-19/probe.py`.

## Integration verification

`ruff check .`, `ruff format --check .`, and `pytest -q` pass (273 tests).
The tests exercise the scaled-tanh density, inverse density, action limits,
and task-specific configuration at the new bound.

A fresh real-Isaac PMD run completed 18 iterations / 4,608 transitions,
including critic warmup and 128 policy projections, with no learner/cadence
overrides. This is an integration smoke test, not a learning result:

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --logger noop --reward_source environment --max_iterations 18 --num_envs 32 \
  --run_name action_bound4_smoke --set runner.validation_interval=0 \
  --set runner.validation_interval_transitions=null
```

Checkpoint:
`logs/train/lift_cube_franka/2026-09-19_09-24-11_action_bound4_smoke/model_18.pt`.

## Interpretation and next check

This shows that the old bound severely disrupts a known working controller,
and that the new range admits reliable lifting in these trials. It does
**not** prove lifting is impossible at bound 1, or that PMD will learn at
bound 4. Clipping also changes the expert's observation history; the
arm-only conditions isolate some, but not all, of those effects.

Widening the bound also broadens initial executed-action exploration because
the latent standard deviation is unchanged. Start a fresh PMD run: loading
old actor weights under a different bound changes their behavior. For an old
run's evaluation, preserve its original bound from its saved configuration.

Next, measure a frozen PMD critic's action rankings against sampled returns
before changing projection cadence. Keep reward learning out of that check
so reward drift is not an additional variable.
