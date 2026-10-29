# Ten-demo successor-PMD fine-tuning

> This is the original screen protocol using the MSE actor with latent std
> 0.1. That screen was not launched. The current mean-preserving spread
> initialization and its first-batch results are in
> [spread_calibration.md](spread_calibration.md); use that checkpoint rather
> than the MSE actor named below for any new fine-tuning run.

This screen uses the broader Lift reset (cube XY half-range **0.06 m**), not the
original narrow Easy Lift task. Ten complete successful demonstrations, one
fully fitted BC actor, and identical data/compute budgets are shared by all arms.

## Initialization and data

Study artifacts: `logs/benchmarks/lift_xy06_bc_scratch_20260923/`.
Use **`demos_10.hdf5`** for expert features, BC, and protected replay, and
**`xy06_bc_10d_s42.pt`** for the actor. Do not substitute the 100-demo pool.
The actor completed 40,000 updates; whole-trajectory cross-validation chose
zero L2 over `0, 1e-6, 1e-5`. It is not an early checkpoint selected for failure.

Fresh evaluation (32 attempts each on reset seeds 8101/8102):

| Policy | Successes |
| --- | --- |
| BC, deterministic | 33/64 (51.6%) |
| Same BC, stochastic with its existing latent std 0.1 | 0/64 |

The earlier deterministic qualification was 31/64 on seeds 8001/8002. These are
one trained actor, not independent training-seed replications. Collection noise
is an existing weakness: fine-tuning does **not** start from 50% stochastic
success. Do not silently change std or entropy to hide this discrepancy.

Every online arm loads the same actor **and reference-policy snapshot**, starts
fresh critics/optimizers, and spends **10,000 demo-only critic update calls**
before collecting anything. Both successor and separate entropy critics learn
against the frozen stochastic BC policy. Target networks use the existing
Polyak update. The actor is checked unchanged; no online UTD credit is earned.
`model_0.pt` is the warmed starting state. This larger warm-up budget replaces
the original study's 1,000 only for this screen; it is not an accuracy guarantee.
Demo actions come from the teacher, whereas bootstrap actions come from BC:
small demo TD loss does not establish accuracy on off-demonstration actions.
Stochastic independent evaluation includes sampled-return value diagnostics.

## What the weights actually mean

Let `g_t` be the unit terminal success event, `phi_t` the six raw non-success
features, `w` the learned unit-L2-constrained reward vector, and `dt=0.02`.
Choose nonnegative labels `(lambda_sparse, lambda_IRL, lambda_BC)` summing to one,
with `b = lambda_BC < 1`, and set

```
a = lambda_sparse / (1 - b)
r_t = a * g_t + (1 - a) * dt * dot(w, phi_t)
L_actor = (1 - b) * L_PMD[r] + b * mean_squared_expert_action_error
```

`L_PMD[r]` is the existing projection onto the frozen exponential improvement
target built from the combined reward's Q estimate. Each successor head is
scalarized with the **combined coefficients before the median** is taken.
There is no separate sparse-policy projection or IRL-policy projection.

This is a nested convex mixture (reward mixture, then actor-loss mixture), **not**
an identity `lambda_sparse L_sparse + lambda_IRL L_IRL + lambda_BC L_BC` of three
independent actor losses. Quantiles and exponential targets are nonlinear, and
adaptive target-KL scaling also changes the effective Q coefficient. Equal
numbers do not imply equal gradient sizes or reward magnitudes. BC is action
MSE, not log likelihood. Interpret these as ablation settings, not percentages
of learning attributable to each signal.

Reward fitting still minimizes `w·(mu_collection - mu_expert)` plus its existing
regularization, with projected RMSprop. Its optimizer is separate; do not mix
its scalar loss into the actor's loss. The mixture changes how the actor uses
that fitted reward, not the reward learner's step size or regularization.
Eight reward steps use the latest 32 attempted episodes, then the reward stays
fixed for the complete PMD block. Feature gaps retain their collection-policy
version. No feature rescaling or extra IRL rollouts are introduced.

Entropy remains at rate **0.006**, integrated with dt in the separate entropy
critic and immediate policy term. Reward-mixture coefficients do not rescale
that rate. BC still scales the whole PMD projection relative to its MSE term.

## Initial screen

| Setting | Sparse / IRL reward coefficients | BC weight | Simplex labels (sparse, IRL, BC) |
| --- | --- | --- | --- |
| sparse_bc10 | 1 / 0 | 0.1 | (0.9, 0, 0.1) |
| mixed_bc10 | 0.5 / 0.5 | 0.1 | (0.45, 0.45, 0.1) |
| irl_bc10 | 0 / 1 | 0.1 | (0, 0.9, 0.1) |
| sparse_bc90 | 1 / 0 | 0.9 | (0.1, 0, 0.9) |
| mixed_bc90 | 0.5 / 0.5 | 0.9 | (0.05, 0.05, 0.9) |
| irl_bc90 | 0 / 1 | 0.9 | (0, 0.1, 0.9) |

Sparse/IRL endpoints use existing `environment` / `learned` reward sources;
the middle uses `environment_plus_learned` with
`reward.environment_weight=0.5` and `reward.learned_weight=0.5`.
The new coefficients default to one, preserving the previous additive configs.
BC-only is the frozen zero-online-interaction reference, not another learner.

Initial screen: seed 42, at most **200,000 physical training interactions per
arm**, 32 complete attempted episodes per batch (about 800 trajectories at full
length). Keep gamma=1, H=250, quantile=0.5, UTD=2, 16 improvements per batch,
eight projection steps, and 50% protected-demo replay after online data exists.
UTD counts retained real transitions, not padded steps. Short failures remain
in replay; physical/retained/discarded interactions and trajectories are separate.
At full length: 400,000 online critic update calls plus 10,000 warm-up calls per
arm. Every call trains both value components; it is not one Adam step in total.

Use isolated evaluation processes at initialization and near 80k/200k physical
interactions, with both policy modes and 64 attempted episodes per mode. Eval
data never enters replay or reward fitting. W&B evaluation charts use
`Samples/training_environment_interactions`, not checkpoint index. Evaluation
and warm-up costs are reported separately. Preserve uninterrupted training and
its replay; do not resume a partial learner merely to evaluate it.

The bounded 8k-interaction smoke run precedes the screen and is not a replicate.
Select promising settings by independent success, inspect stochastic stability
and value diagnostics, then decide on longer runs and seeds 43/44. A single-seed
weight screen cannot establish a sample-efficiency advantage.

## Smoke result and current decision

The mixed-reward, BC-weight-0.1 smoke completed: 8,000 physical interactions,
7,819 retained transitions, 32 attempted trajectories, 16 policy improvements.
The saved initialization verifies exactly 10,000 successor and 10,000 entropy
optimizer steps, zero online interactions, and bit-identical BC actor/reference
parameters with an untouched actor optimizer.

Independent evaluation of that initialization reproduced 33/64 deterministic
and 0/64 stochastic successes. After the single training batch, **both modes
were 0/64**. This is a regression, not a successful fine-tuning baseline.
At initialization, entropy start-value predictions were about 2.55 versus
sampled returns 7.30/9.46 across the two reset seeds. Small demo TD loss did not
establish accurate values on the stochastic policy's visited states.

The six-setting workflow is prepared at
`logs/benchmarks/lift_xy06_finetune_20260923/workflow.py` but **has not been
launched**. The next discussion is distributional BC (negative log likelihood)
with controlled nonzero spread: current MSE BC does not train the std head,
and its preset exploration already destroys success before fine-tuning.
Do not interpret this one smoke as establishing that likelihood BC alone will
fix the subsequent value error or policy drift.
