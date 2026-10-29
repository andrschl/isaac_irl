# Likelihood BC before fine-tuning

The ten-demo MSE actor scored 33/64 deterministic but 0/64 stochastic successes
on broader-reset Lift. MSE fitted its mean but left latent std at 0.1. The
first mixed-reward fine-tuning smoke then lost deterministic success too.
Before the reward/BC-weight sweep, qualify a distributional BC initialization.

## Objective and spread

For expert action `a_E`, the bounded actor uses
`u ~ Normal(mu(s), diag(sigma(s)^2))`, `a = 4*tanh(u)`. Fit

```
L_BC = -mean(log pi(a_E | s))
sigma(s) = sigma_min + (sigma_max - sigma_min) * sigmoid(z(s))
```

Log density includes the existing tanh/action-bound Jacobian and sums over
action dimensions. The NLL objective can be negative (it is a continuous
density); deterministic action MSE remains a separate diagnostic.
The std head is trainable. Its initial bias gives exactly the configured
initial std, here 0.1, with ceiling 2.0. The floor is a **latent** std, not
action-space std. Smooth bounding avoids a hard clamp's exactly flat region,
though sigmoid gradients can still become small near a bound.

Use `algo.pmd.policy.std_parameterization=smooth_bounded`; the default remains
`clamped_log` for previous experiments. Distinct head keys reject silent
reinterpretation of old checkpoint weights. Smooth checkpoints record their
bounds and reject loading with a different floor/ceiling. Use the same policy
config for offline fit, evaluation, value warm-up, and online fine-tuning.
No automatic conversion of an old MSE actor is performed.

Later RL entropy remains at rate 0.006 with the existing dt scaling and separate
entropy critic. It is not an offline anti-collapse guarantee, so the positive
floor is enforced throughout. Reward learning, TD targets and PPO are unchanged.

## Selection, not early stopping

Use precisely the existing `demos_10.hdf5` from
`logs/benchmarks/lift_xy06_bc_scratch_20260923/`, on cube XY half-range 0.06 m.
Five whole-trajectory folds hold out two demonstrations each. Fit every
candidate from scratch for 40,000 Adam steps, batch 256, cosine learning rate
3e-4 to 3e-6. Compare latent std floors **0.001, 0.005, 0.02** and mean-network
L2 coefficients **0, 1e-5**. L2 retains its previous definition (mean-network
weight matrices only); the spread safeguard is the selected positive floor.

Select by the average **final held-out NLL**, never by simulator success or an
early checkpoint. Each selected floor/L2 pair is refitted on all ten demos;
the best cross-validated pair supplies the final actor. No holdout is removed
from that final fit, online BC, or protected replay. Full-data train/validation
NLL, MSE, std range and objective traces are logged to W&B.

Example for one floor:

```bash
python scripts/validate_bc.py --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --expert_data_path logs/benchmarks/lift_xy06_bc_scratch_20260923/demos_10.hdf5 \
  --loss_type nll --gradient_steps 40000 --learning_rate 3e-4 \
  --final_learning_rate 3e-6 --l2_candidates 0 1e-5 --folds 5 \
  --set env.lift_cube_xy_half_range=0.06 \
  --set algo.pmd.policy.std_parameterization=smooth_bounded \
  --set algo.pmd.policy.minimum_std=0.005 \
  --output logs/bc/nll_floor005.pt
```

After selection, evaluate the final NLL actor and unchanged MSE reference on
matched fresh reset seeds 8201/8202: 64 attempted episodes per policy mode,
both deterministic and stochastic. Evaluation is separate from fitting, its
interactions are counted separately, and its data never enters optimization.
There is no target-success-band selection. This remains one training seed.

The bounded workflow and status live in
`logs/benchmarks/lift_xy06_nll_bc_20260923/`. It runs offline BC and independent
evaluation only, then stops for review. A successful qualification would allow
demo-only value warm-up and a new NLL-BC fine-tuning screen. Retune online BC
weights: summed-action NLL and mean-action MSE do not share the same scale.

## Completed result (2026-09-23)

All three final NLL actors and the original MSE actor were evaluated on the
same fresh reset seeds, 8201 and 8202. Each mode used 32 complete attempted
episodes per seed. No evaluation transition entered training.

| Actor | Five-fold held-out NLL | Deterministic success | Stochastic success |
| --- | ---: | ---: | ---: |
| NLL, std floor 0.001 | −30.30 | 0/64 | 0/64 |
| NLL, std floor 0.005 | −21.48 | 0/64 | 0/64 |
| NLL, std floor 0.02 | −13.75 | 11/64 | 4/64 |
| Original MSE actor | not comparable | 35/64 | 0/64 |

The lower-is-better NLL criterion preferred floor 0.001, while the largest
floor had the best NLL-actor task performance. The original MSE actor remains
much stronger in deterministic control. Its stochastic policy lifted the cube
in 20/64 episodes but never satisfied the final hold rule. Full-data training
mean-action MSE was about 0.0260, 0.00817, 0.00281 for the three NLL floors,
versus 0.000070 for the original MSE fit. The likelihood objective allowed
larger action-mean errors while adjusting the spread. These data do not support
starting the planned PMD sweep from the NLL-selected actor.

The machine-readable comparison is
`logs/benchmarks/lift_xy06_nll_bc_20260923/all_floors_evaluation.json`.
Each actor evaluation cost 32,000 simulator interactions; online training cost
remains zero. A next experiment can preserve the fully fitted MSE mean policy
and fit only its conditional spread by likelihood, then evaluate that combined
actor before value warm-up and fine-tuning.
