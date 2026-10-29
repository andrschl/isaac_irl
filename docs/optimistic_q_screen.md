# Optimistic successor-Q control on ten-demo Lift

This is the direct test of *ensemble optimism*, distinct from the
[higher-entropy actor screen](aggressive_lift_tuning.md). The latter changed
the entropy rate and actor spread bound; it did **not** test optimistic Q
reduction. This screen changes only the successor-control quantile and keeps
entropy at its original `0.006` rate.

## Control score and comparison

For each candidate action, scalarize all three successor heads under the
**same** fixed sparse terminal-success reward, then take the configured
quantile across their three scalar Q values. If sorted values are
`q1 <= q2 <= q3`, the tested controls are:

| Quantile | Control score | Role |
| ---: | --- | --- |
| `0.5` | `q2` | Existing median baseline |
| `0.9` | `0.2 q2 + 0.8 q3` | Near-maximum optimism |
| `1.0` | `q3` | Maximum-head optimism |

The percentage is across *scalarized* heads and uses linear interpolation.
Successor TD targets, ensemble training, target networks, and the separate
entropy critic do not change. The actor is steered toward actions valued by
the upper ensemble tail; this is only useful if head disagreement identifies
valuable unknown actions rather than approximation error.

Reuse the [success-aligned sparse RL + BC screen](success_aligned_irl_screen.md)
exactly: seed 42, the same ten demonstrations and calibrated BC actor, BC
weight `0.99`, 10,000 demo-only updates for both value components, 32 complete
episodes/batch, UTD 2, one PMD improvement/batch, actor learning rate `3e-6`,
unchanged entropy rate `0.006`, five batches, and a 40,000-training-
interaction budget per run. Its `0.5` sparse run is the existing matched
baseline; the new runs use only
`--set algo.pmd.value.successor.control_quantile=0.9` or `=1.0`.

Save every post-update checkpoint. Evaluate `model_1.pt` through `model_5.pt`
in a separate Isaac process on seeds 8501/8502, 32 attempted episodes per
seed and policy mode, with the same environment and actor configuration.
Report deterministic and stochastic success against physical *training*
interactions, plus evaluation interactions separately. The two new runs cost
up to 80,000 physical training interactions / 320 attempted training episodes;
their 10 checkpoint evaluations cost up to 320,000 additional simulator
interactions / 1,280 attempted evaluation episodes. Evaluation never enters
replay. Compare the realized actor KL, action std, Q-head disagreement, and
impossible sparse-reward Q predictions above one alongside success.

Advance an optimistic quantile to a mixed sparse+IRL test only if a checkpoint
reaches at least 38/64 stochastic successes without dropping below 32/64
deterministic successes. This is a screen, not a performance claim: selection
from five checkpoints and one training seed needs fresh evaluation and
training seeds before any sample-efficiency conclusion. If neither passes,
stop the interaction sweep and diagnose whether the upper Q head is ranking
actions or merely amplifying critic error.

## Results (2026-09-25)

Both optimistic runs completed all five batches: 40,000 physical training
interactions and 160 complete attempted training episodes each. The saved
`model_0.pt` actor, reference policy, and online and target successor/entropy
critics were bit-for-bit identical across the two runs and the prior median
baseline. This checks their common BC and demo-only critic warm start. All
checkpoints below were evaluated independently on the same two seeds; each
cell is successes out of 64 attempted episodes. Evaluation interactions were
never added to replay.

| Training interactions | Median `0.5`, det/stoch | Optimistic `0.9`, det/stoch | Max `1.0`, det/stoch |
| ---: | ---: | ---: | ---: |
| 8,000 | 32/33 | 32/32 | 32/33 |
| 16,000 | 32/33 | 32/30 | 32/33 |
| 24,000 | 32/32 | 32/33 | 32/32 |
| 32,000 | 32/33 | 32/32 | 32/31 |
| 40,000 | 32/33 | 31/31 | 32/32 |

Here `det/stoch` means deterministic/stochastic successes, each out of 64;
the zero-online-interaction BC reference was 32/33. Neither optimistic
setting reached the predeclared 38/64 stochastic gate, so no mixed-IRL or
seed-43/44 follow-up was launched. The two new arms used 80,000 training
interactions in total. Their ten independent checkpoint evaluations used
another 320,000 simulator interactions (1,280 attempted episodes), reported
separately; the existing median baseline was not retrained.

The upper-tail score did not produce a useful success gain under this
conservative PMD schedule. Final collection-policy KL was about `0.023`
for `0.9`, `0.014` for `1.0`, and `0.015` for the median baseline, so a
larger control quantile did not consistently produce more policy movement.
Some start-state sparse-Q estimates were negative even though the true
finite-horizon sparse return is in `[0, 1]`; this occurred in the median
baseline too. It is evidence of critic error, **not** proof that optimism
caused the error. In this one-seed screen, taking the high head did not
overcome it. The next useful investigation is value/action ranking and
critic calibration, not more entropy or a larger optimism sweep.

W&B: [median training](https://wandb.ai/andrschl/isaac_irl/runs/lpqx1mrv),
[median per-batch evaluation](https://wandb.ai/andrschl/isaac_irl/runs/9kv3m4pk),
[`0.9` training](https://wandb.ai/andrschl/isaac_irl/runs/q0sfb08r),
[`0.9` per-batch evaluation](https://wandb.ai/andrschl/isaac_irl/runs/wt6zj9wy),
[`1.0` training](https://wandb.ai/andrschl/isaac_irl/runs/ey5q57ys), and
[`1.0` per-batch evaluation](https://wandb.ai/andrschl/isaac_irl/runs/y3iv8yvp).
The corresponding independent-evaluation JSON files live under
`logs/benchmarks/lift_optimistic_q_screen_20260925/`.
