# Calibrated-BC PMD actor-cadence screen

This is a two-arm follow-up to [the ten-demo weight screen](ten_demo_weight_ablation.md).
It tests whether the repeated one-step/backtracked PMD projections and large
within-batch policy drift are limiting fine-tuning. Do not change reward
weights, critic architecture, or the replay policy during this screen.

Both arms use the same ten demonstrations, the calibrated spread-0.005 BC
actor, sparse terminal success reward, online BC-MSE weight 0.99, seed 42,
32 complete episodes per collection, 50% protected-demo replay, UTD 2,
10,000 demo-only warm-up updates for each value critic, and five collection
batches. Thus each arm gets at most 40,000 training interactions / 160 new
trajectories. The actor learning rate is `3e-6` in both arms; only
`algo.pmd.policy_improvements_per_batch` differs: **1** or **4**. Critic update
counts remain 16,000 per full batch in either arm; each policy improvement
gets the complete intervening critic block. The configured projection maximum
remains eight actor optimizer steps.

Inspect actual projection steps, backtracks, per-projection KL, cumulative KL
from the collection policy, BC loss, and same-policy sampled-return value
diagnostics. A lower learning rate is only useful if it changes the realized
projection behavior; do not infer update size from the loss coefficient alone.

Evaluate the common warm-start once, and each arm after its first and fifth
batches, using reset seeds 8601/8602 with 16 attempted episodes per seed and
policy mode. That is 32 episodes per mode per checkpoint. Evaluation remains
external and never enters replay. At full episode length, the planned cost is
80,000 training plus 80,000 evaluation physical interactions across the
two arms, excluding previously collected demonstrations and expert training.
Treat this as a screen: replicate any apparent gain with fresh training seeds
and more evaluation episodes before claiming sample-efficiency improvement.

## Results

The initial actor and both critics were bit-identical between arms after
demo-only warm-up. Each arm retained 40,000 of 40,000 physical training
transitions from 160 complete episodes, made 90,000 updates of each critic
(10,000 warm-up + 80,000 replay updates), and used no reward-learner updates.

| Training interactions | One improvement: deterministic / stochastic | Four improvements: deterministic / stochastic |
| ---: | ---: | ---: |
| 0 | 25 / 24 | same initial checkpoint |
| 8,000 | 24 / 24 | 24 / 24 |
| 40,000 | 24 / 24 | 24 / 23 |

Each count is successes out of **32** attempted episodes per mode, on reset
seeds 8601/8602. Neither arm improved over the common BC initialization.
These seeds gave the BC actor 78% deterministic / 75% stochastic success,
substantially higher than the roughly 50% observed on earlier reset seeds;
the small paired evaluation is suitable for this screen, not a precise
population success estimate. The two arms together used 80,000 training and
80,000 separate evaluation physical interactions, with 320 training and 320
evaluation trajectories. Evaluation data were never inserted into replay.

Reducing actor LR from `3e-4` to `3e-6` did repair the projection mechanism:
the one-improvement arm made five policy improvements using **37 actual actor
optimizer steps**, ending with collection-policy KL 0.015. The four-improvement
arm made 20 improvements using **126 actor optimizer steps**, ending with
collection-policy KL 0.110. By contrast, the previous 16-improvement runs took
one actor step per improvement with roughly 6–7 backtracks and had final-batch
collection-policy KL 4.6–8.4. This follow-up did **not** establish a
success gain from the repaired update geometry.

The critic remains a concern. At zero interactions, the sparse start-value
prediction was approximately −0.04 while sampled success returns on these
evaluation episodes averaged 0.625 and 0.875 across the two reset seeds. At
40k, the one-improvement arm predicted 0.635/0.799 against sampled
0.625/0.875; the four-improvement arm predicted 1.214/1.857 against sampled
0.625/0.812. A finite-horizon sparse return pays at most one, so the latter
estimates cannot be calibrated sparse values. The final critic also lags the
last actor projection, so the sampled-return comparison is a diagnostic, not
an exact conditional-Q error estimate. Low one-step TD loss is not enough to
clear the actor for aggressive improvement.

The [one-improvement training](https://wandb.ai/andrschl/isaac_irl/runs/575hm6ur)
and [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/cw5v1ln0), and the
[four-improvement training](https://wandb.ai/andrschl/isaac_irl/runs/0n2farnc)
and [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/324h5vt2), contain
the detailed metrics. JSON results are under
`logs/benchmarks/lift_xy06_actor_cadence_20260923/`.
