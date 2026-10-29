# Calibrated-BC Lift fine-tuning weight screen

Use the ten-demo, fully fitted MSE-mean plus calibrated-spread actor from
[spread_calibration.md](spread_calibration.md). The policy has a smooth-bounded
latent standard deviation with floor 0.005. All arms load this same actor into
both the live and reference policies and make 10,000 demo-only updates of each
value component before collecting any new experience. Online BC is action MSE.

The coefficient `lambda` controls the existing nested actor objective:

```
L_actor = lambda * L_PMD[r] + (1 - lambda) * L_BC-MSE
```

For the sparse arm, `r = g`. For the mixed arm,
`r = 0.5*g + 0.5*dt*dot(w, phi)`. Here `g` pays one unit once for terminal
success, and the reward learner fits `w` against each frozen collection
policy's complete-episode feature expectations. The successor heads scalarize
the complete reward before their median reduction; entropy retains its
separate critic and rate 0.006. The two reward components are **not** separate
actor losses, and `lambda` does not determine their gradient-norm ratio.

| Arm | `lambda` | `bc.weight` | Sparse reward | Learned reward |
| --- | ---: | ---: | ---: | ---: |
| sparse_50 | 0.5 | 0.5 | 1 | 0 |
| sparse_10 | 0.1 | 0.9 | 1 | 0 |
| sparse_01 | 0.01 | 0.99 | 1 | 0 |
| mixed_50 | 0.5 | 0.5 | 0.5 | 0.5 |
| mixed_10 | 0.1 | 0.9 | 0.5 | 0.5 |
| mixed_01 | 0.01 | 0.99 | 0.5 | 0.5 |

Use seed 42, the same ten demonstrations and protected replay, the broader
0.06 m cube reset, and a maximum of 40,000 physical training interactions
(five complete batches of 32 episodes) per arm in the initial screen.
Independent evaluation uses reset seeds 8501/8502, 32 episodes per seed and
policy mode, at 0, 8,000 and 40,000 training interactions. Evaluation data
never enters replay. Run one Isaac process at a time on the GPU. Select any
extension by independent stochastic success and stability, not by training
reward or in-sample success. The two earlier one-batch mixed runs at BC weights
0.1 and 0.9 are diagnostics only; their data are not reused for this screen.

The source configuration is `configs/lift_cube_franka/sparse_irl_bc.yaml`.
Each run overrides the calibrated policy settings, common data paths,
`bc.weight`, reward source and (for mixed arms) its two coefficients. The
training launcher saves the resolved configuration beside every checkpoint.
Keep standard PPO unchanged.

## Completed five-batch screen (seed 42)

Each cell is deterministic / stochastic terminal successes out of 64
independent attempted episodes (reset seeds 8501/8502). The shared BC starting
policy was 32/64 / 33/64 at zero online training interactions.

| Arm | 8,000 training interactions | 40,000 training interactions |
| --- | ---: | ---: |
| sparse_50 | 19 / 25 | 24 / 21 |
| sparse_10 | 24 / 30 | 0 / 0 |
| sparse_01 | 14 / 19 | 33 / 32 |
| mixed_50 | 13 / 18 | 24 / 25 |
| mixed_10 | 16 / 24 | 6 / 11 |
| mixed_01 | 10 / 18 | 23 / 18 |

No arm surpassed the BC initialization at the fixed endpoint. Sparse_01
recovered approximately to baseline after an initial drop; this is not evidence
that online training improved success. The mixed arms did not outperform their
sparse counterparts at the same PMD weight on these reset seeds. These are
single-training-seed screening results, not replicated estimates of an
algorithm's sample efficiency. The success of the final collected batch belongs
to the **pre-update collection policy**, not the saved post-update checkpoint;
for example, mixed_10's fifth collection succeeded 21/32 but its final
checkpoint evaluated at 6/64 deterministic and 11/64 stochastic.

Every arm retained all 40,000 physical training transitions from 160 complete
attempted episodes. Each made 80 policy improvements, 90,000 successor and
90,000 separate entropy-critic optimizer steps (10,000 demo-only warm-up plus
80,000 replay-based steps); each mixed arm made 40 reward optimizer steps.
Across six arms that is **240,000 training interactions and 960 training
trajectories**. The three checkpoint evaluations per arm used another
**576,000 physical interactions and 2,304 evaluation trajectories**, never
inserted into training. Expert collection and expert PPO training are separate
costs. Because evaluation already exceeded online training cost, do not add
more checkpoint evaluations casually in a real-robot sample-budget comparison.

There is a concrete optimizer diagnostic behind the poor tradeoff: every
saved actor optimizer took exactly 80 steps for 80 improvements, although
`projection_steps=8`. The final W&B summaries report one projection step and
roughly 6–7 parameter backtracks per improvement. The raw Adam proposal is
therefore repeatedly much larger than the permitted KL step. Even at
`lambda=0.01`, the final projection KL is about 0.028 per improvement. Adam is
largely insensitive to uniform gradient scaling, so lowering `lambda` alone
need not proportionally reduce its parameter step when the BC gradient is
small. This is a mechanism to test, not proof that it is the only cause.
Independent sampled-return diagnostics also show value error: for sparse_01
at 40k, the start-state sparse-reward Q predicts about 1.14 while sampled
returns average about 0.5; at initialization, the entropy Q predicts about
−23 versus sampled returns about −107. Sampled returns are noisy estimates of
conditional values, but the systematic gap cautions against equating low TD
loss with accurate policy evaluation.

The saved JSON results are under
`logs/benchmarks/lift_xy06_calibrated_weights_20260923/`. W&B contains the
[sparse_01](https://wandb.ai/andrschl/isaac_irl/runs/3pdnprjt),
[sparse_10](https://wandb.ai/andrschl/isaac_irl/runs/wezqmh5e),
[sparse_50](https://wandb.ai/andrschl/isaac_irl/runs/o0bz7gm5),
[mixed_01](https://wandb.ai/andrschl/isaac_irl/runs/1kswu9fr),
[mixed_10](https://wandb.ai/andrschl/isaac_irl/runs/sirrnxj8), and
[mixed_50](https://wandb.ai/andrschl/isaac_irl/runs/izmznuza) training runs;
each has a separate evaluation run in the same project. A sensible next
controlled test is a smaller actor learning rate with this exact warm-start,
reward, BC weight, and interaction budget, checking realized projection steps
and independent success before changing the reward objective again.
