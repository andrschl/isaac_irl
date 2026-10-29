# Properly fitted BC baseline

This supersedes success-band checkpoint selection in `bc_calibration.md`.
BC is fitted for generalization, not deliberately weakened to leave room for RL.
The ten demonstrations, task, actor architecture and action bounds stay fixed.

## Objective and selection

For expert observations/actions `(o, a)`, fit the deterministic bounded mean
action `m_theta(o)` using

\[
L(\theta)=\mathbb{E}_{(o,a)\sim D_{train}}
\left[\frac{1}{A}\lVert m_\theta(o)-a\rVert_2^2\right]
+\frac{\lambda}{2}\sum_{W\in\mathcal W}\lVert W\rVert_F^2.
\]

`A` is action dimension. `W` contains the trunk and mean-head weight matrices;
biases and the exploration head are not penalized. This is explicit L2 in the
Adam objective, not decoupled AdamW weight decay. MSE leaves the initially
constant latent standard deviation unchanged at 0.1.

- Shuffle episode indices once using seed 42. Five folds each fit eight whole
  demonstrations and validate on the other two. Each demo is held out once.
  Never split adjacent transitions between fitting and validation.
- Compare `lambda = 0, 1e-6, 1e-5`, with matched initialization and minibatch RNG
  across candidates. Use the same 40,000 Adam steps, batch 256, and cosine
  learning-rate decay from `3e-4` to `3e-6` for every fit.
- Record full training and held-out action MSE throughout. Choose lambda by
  the **final** mean held-out MSE across folds, not an intermediate checkpoint
  or simulator success target. Inspect curves for unresolved optimization or
  overfitting; reaching a step budget is not itself proof of convergence.
- Refit from scratch on **all ten demos** using the chosen lambda and the
  same complete schedule. Validation folds do not remove demos from the final
  baseline, online BC, IRL expert set or protected demonstration replay.
- When a nonzero lambda wins, also fit a matched all-demo unregularized control
  to separate the effects of the learning-rate schedule and regularization.

All of this is offline. W&B logs use `BC/optimizer_steps`, never environment
steps. Fold identities, learning curves, scores and selection travel with the
saved artifacts. No evaluation transitions enter fitting.

## Reproduce

Use a new output path; the validation launcher refuses to overwrite a study.

```bash
python scripts/validate_bc.py --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml --seed 42 \
  --expert_data_path logs/demos/lift_sparse_qualify/demos.hdf5 \
  --gradient_steps 40000 --learning_rate 3e-4 --final_learning_rate 3e-6 \
  --l2_candidates 0 1e-6 1e-5 --folds 5 \
  --output logs/benchmarks/new_bc_regularized/bc_s42.pt
```

For a single fit with a known lambda, `scripts/train_bc.py` accepts
`--l2_weight`, `--final_learning_rate`, and an optional whole-episode
`--validation_fraction` for diagnostics. Keep that fraction zero for final fits.

Qualify the selected all-demo actor deterministically in a fresh simulator
process, on reset seeds not used in the earlier calibration. Evaluate a fitted
unregularized control separately if available. Report success and evaluation
interaction cost; do not select a checkpoint to hit 30–60% success. A stronger
baseline is welcome. Stochastic collection is a separate qualification.

Regularization cannot guarantee generalization from ten trajectories. Held-out
MSE is a supervised diagnostic, not a guarantee of closed-loop stability, and
five folds over ten demos are still a small, overlapping training-set study.

## First completed study — 2026-09-22, actor seed 42

All fifteen fold fits and the final all-demo fit completed 40,000 updates.
The best tested coefficient was **lambda = 0**; adding L2 did not help this
demonstration set under the matched schedule. Do not force regularization when
it worsens held-out prediction, or interpret this limited grid as ruling out
all other regularizers or strengths.

| L2 coefficient | Mean training MSE across folds | Mean held-out MSE |
| --- | ---: | ---: |
| 0 | 1.9185e-5 | **6.3450e-5** |
| 1e-6 | 2.7047e-5 | 7.8609e-5 |
| 1e-5 | 5.2920e-5 | 1.2456e-4 |

For lambda zero, mean held-out MSE at 36k/38k/40k was
6.3731e-5 / 6.2888e-5 / 6.3450e-5: roughly stable late in fitting, not evidence
of a substantial late deterioration. The final actor uses all 2,500 expert
pairs and has training MSE 2.0800e-5. This is not a proof of no overfitting.

Independent deterministic evaluation in a fresh Isaac process produced
**52/64 successes (81.25%)**: 28/32 on reset seed 5001 and 24/32 on seed 5002.
This exceeds the old requested success band; it is retained as the result of
proper fitting. The old 47% checkpoint is no longer the recommended baseline.
The earlier comparison used different reset seeds, so the success difference
is not a paired causal estimate of the schedule's benefit.

- Actor: `logs/benchmarks/lift_bc_regularized_20260922/bc_s42.pt`.
- Fold records, selection and full-fit trace: sibling `bc_s42_validation/`.
- Evaluation: sibling `evaluation_5001_5002/bc_s42.json`.
- [Full-fit W&B](https://wandb.ai/andrschl/isaac_irl/runs/t4a2f9si).
- [Independent evaluation W&B](https://wandb.ai/andrschl/isaac_irl/runs/vb98ongg).

Cost: 640,000 offline optimizer steps total (600k validation fits + 40k final
fit); **zero online training interactions**. Evaluation used 16,000 physical
interactions, 64 attempted full trajectories, no discarded transitions and no
genuine failures. Twelve episodes failed the terminal success criterion.
Evaluation data never entered optimization. No additional demos were recorded.

Stochastic collection with latent std 0.1 and actor seeds 43/44 remain
unqualified. No new RL/IRL training or critic warm-up was run in this BC study.
The regression suite passes 456 tests, including validation isolation,
exploration-head preservation, exact L2 gradients, and full-budget refitting.

The subsequent [demo-count screen](bc_demo_count.md) reduces training data
while keeping this full-fit recipe fixed; it does not shorten fitting to tune
the starting success rate.
