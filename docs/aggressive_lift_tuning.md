# More aggressive ten-demo Lift fine-tuning screen

The [success-aligned screen](success_aligned_irl_screen.md) was flat on fixed
evaluation seeds after all five batches. Its BC weight of 0.99, one PMD
improvement per batch, and fitted action std near 0.005 made it a conservative
control. This follow-up tests actor update strength and controlled exploration,
not a new reward or value architecture.

## Matched protocol

Use the same ten demonstrations, calibrated BC actor, seed 42, Easy Lift
without curriculum, 32 complete training episodes per batch, 50% protected
demonstration replay, finite horizon 250, gamma 1, UTD 2, 10,000 demo-only
updates of both critics, actor learning rate `3e-6`, and five batches as the
prior screen. Save every post-update checkpoint. All arms use the successor
backend, median control quantile, separate entropy critic, and BC action MSE.

| Run name | Policy reward | BC weight | PMD improvements/batch | Entropy rate | Spread KL cap |
| --- | --- | ---: | ---: | ---: | ---: |
| `sparse_bc90_p4_s42` | fixed terminal +1 | 0.90 | 4 | 0.006 | 0.002 |
| `sparse_bc90_p4_entropy12_s42` | fixed terminal +1 | 0.90 | 4 | 0.012 | 0.010 |
| `mixed_bc90_p4_s42` | fixed terminal +1 plus exact four-feature IRL | 0.90 | 4 | 0.006 | 0.002 |

The first two compare an entropy/spread change at a fixed sparse reward and
actor schedule. They do **not** test ensemble optimism; that separate,
one-variable follow-up is the [optimistic successor-Q screen](optimistic_q_screen.md).
The third asks whether the learned task features help under that schedule.
The earlier `0.99`-BC, one-improvement arm is the conservative
reference; the older [four-improvement cadence screen](actor_cadence_screen.md)
already tested `0.99` BC with this learning rate, so no duplicate training
control is needed. The action-std lower bound remains the calibrated `0.005`:
the [spread calibration](spread_calibration.md) found that increasing it to
`0.02` reduced stochastic BC success. The exploration arm instead changes the
entropy incentive and the spread portion of the actor trust region together;
it is a named setting, not a one-variable claim about either component.

Each complete run costs at most 40,000 physical training interactions and
160 attempted trajectories, with 80,000 online optimizer steps for each critic
and 20 policy improvements. The IRL arm takes five exact reward solves and no
reward optimizer steps. Training-batch success is **not** model selection:
each batch contains different reset draws and was collected before that
batch's policy update.

Evaluate `model_1.pt` through `model_5.pt` externally on seeds 8501/8502,
32 episodes per seed and policy mode, using the same environment and policy
overrides as training. That gives 64 deterministic and 64 stochastic attempts
per checkpoint, on fixed reset seeds. Plot W&B evaluation success against
`Samples/training_environment_interactions` (8k, 16k, 24k, 32k, 40k), and
count evaluation interactions separately. No evaluation episode enters replay
or reward fitting. For the three-arm screen, this adds at most 480,000
evaluation interactions; this is simulator diagnostic cost, not online
training data. Compare realized collection-policy KL, per-projection KL,
backtracks, actor optimizer steps, action std, BC loss, and value diagnostics
alongside success.

Advance a setting only if a checkpoint reaches at least 38/64 stochastic
successes without dropping below 32/64 deterministic successes on these
seeds. This is a *screening* threshold, not a performance claim; selecting a
checkpoint from five points requires fresh evaluation seeds and training seeds
before claiming improvement. Stop if none clears it rather than spending a
larger interaction budget on a flat setting.

## Results (24 September 2026)

All three runs completed five batches: 40,000 physical training interactions
and 160 complete attempted training episodes each (120,000 interactions and
480 episodes total). Each received 10,000 demo-only plus 80,000 online updates
of both the successor and separate entropy critic, and made 20 PMD
improvements. Actual actor Adam steps were 140, 134, and 70 in table order.
The mixed arm made five exact reward solves and zero reward-optimizer steps.
Each arm's five separate checkpoint evaluations cost 160,000 simulator
interactions and 640 attempted evaluation episodes; the three-arm total was
480,000 evaluation interactions / 1,920 episodes, none used for training.

Each entry below is successes out of 64 fixed-seed attempted episodes at
8k/16k/24k/32k/40k *training* interactions. The BC reference was 32/64
deterministic and 33/64 stochastic on these seeds.

| Arm | Deterministic after batches 1–5 | Stochastic after batches 1–5 | W&B |
| --- | --- | --- | --- |
| Sparse, BC .90, 4 PMD | 32, 28, 30, 32, 32 | 33, 25, 29, 32, 33 | [train](https://wandb.ai/andrschl/isaac_irl/runs/meo38te8) / [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/qw1nzih4) |
| Sparse, BC .90, 4 PMD, extra entropy/spread | 32, 32, 31, 27, 32 | 32, 34, 31, 27, 31 | [train](https://wandb.ai/andrschl/isaac_irl/runs/ukmo9gsm) / [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/4egeys7b) |
| Sparse + four-feature IRL, BC .90, 4 PMD | 32, 32, 30, 32, 25 | 32, 32, 32, 32, 25 | [train](https://wandb.ai/andrschl/isaac_irl/runs/ap3k785r) / [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/jwit9dzp) |

Per-checkpoint JSON is under
`logs/benchmarks/lift_aggressive_screen_20260924/<run-name-without-_s42>_eval/`.
All three five-point W&B evaluation histories were checked to use physical
training interactions on the x-axis. None reached the 38/64 screening gate;
seeds 43/44 and a still weaker BC weight were **not** run. In particular, the
mixed arm's final checkpoint lost seven deterministic and eight stochastic
successes versus BC on the same evaluation seeds. The repeated batch-4
training-success spike did not predict an independent evaluation peak.

The stronger schedule did move the actors: final collection-policy KL was
approximately 0.145, 0.245, and 0.301, compared with about 0.015 for the
earlier one-improvement sparse control. But action std remained about
0.005–0.012; final spread-specific projection KL in the extra-entropy arm was
below `1e-6`, far under either spread cap. Relaxing that cap was therefore not
the active exploration mechanism. The final sparse-reward start-Q predictions
in the ordinary BC-.90 arm averaged about 1.47 and 1.51 across evaluation
seeds, even though a finite-horizon terminal-success return is bounded by 1.
This is a concrete value-bound violation, not evidence of useful action
ranking. Sampled-return error is not exact conditional-Q error, and the saved
critic fits the pre-projection policy; diagnose action ranking and value
generalization before increasing actor pressure again.

The first mixed-arm evaluation attempt stopped before producing data because
the new checkpoint-basis guard compared a CUDA-loaded scale tensor with a CPU
model scale tensor. Both evaluation and reward-learner restore now compare on
CPU, with a CUDA regression test. The successful retry produced the five
points above; no evaluation transition from the failed attempt was used.
