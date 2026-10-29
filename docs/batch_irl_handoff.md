# Batch IRL implementation status

Status: **implemented and unit-tested; no experiment has been run.** The
protocol itself has moved to [docs/batch_irl_protocol.md](batch_irl_protocol.md),
which is the document to read first. This file is the running record of what is
done and what is not.

## State of the tree

- `python -m pytest -q`: **all tests pass** (423 at the time of writing).
- `ruff check` and `ruff format --check` are clean over `src`, `scripts`, `tests`.
- Python: `/home/andreas/miniconda3/envs/isaac_irl/bin/python`.
- The large pre-existing dirty worktree is intact; nothing was reset or reverted.

## What was verified in real Isaac (2026-09-22)

Short runs of every stage completed cleanly on the simulator:

- **The sparse task.** Two cycles of `sparse_bc.yaml` at 4 environments. The
  reward manager holds seven terms, the six dense ones at weight zero and
  `terminal_success` at one. The `episode_context` observation term evaluates
  both at observation-manager construction (before the reward manager exists)
  and twice per step. The pre-reset tap's own checks passed: the extra
  observation pass has no side effects, and the tapped features reconstruct
  `env.step`'s reward exactly.
- **Expert qualification: 64 of 64.** One recorder wave against the candidate
  PPO teacher qualified every attempt under the final-25-step hold rule, with
  the largest executed action at 3.11 of the +/-4 bound. The bar was 95%.
- **Recording.** Ten complete successful episodes at
  `logs/demos/lift_sparse_qualify/demos.hdf5`, 250 steps each, 38-dimensional
  observations (36 native plus the two episode-clock columns) and 8-dimensional
  actions. Every episode's total environment reward is exactly `1.0`, and the
  stored `terminal_success` feature is `50.0 = 1/dt` on the final step, so the
  `g/dt -> dt * w.phi` round trip is exact rather than approximately right.
- **Loading.** That real file round-trips through the replay loader, on the
  full seven-column basis and on the six-column learned basis, with
  `bootstrap_valid` false at the horizon.
- **Offline BC.** 1,000 steps over the 2,500 recorded pairs took the action MSE
  to 9.2e-4.
- **Both IRL arms**, warm-started from that actor, with demonstration-seeded
  replay: the feature gap is finite and decreasing, the online BC loss starts at
  1.2e-3 (so the warm start is really the actor), and the learned reward has six
  columns in both arms - the sparse coefficient is never learned.

Two real defects surfaced here rather than in a unit test:

1. The successor backend sized its output scale by `1/(1-gamma)`, which is
   infinite at the `gamma = 1` this study uses. An undiscounted successor sum is
   bounded by the episode instead, so the estimator now takes the horizon
   (`successor_return_span`). Discounted behaviour is unchanged, algebraically
   and by test.
2. The recorder reported failed attempts and surplus successes as one
   "discarded" count. At a quota of ten from a wave of 64 that reads as a
   teacher failing 54 times, when in fact it succeeded 64 times. `RecordingCost`
   now separates them and reports the success rate.

## What has not been done

These need the simulator and hours of GPU time, and none of them should be
started without deciding to spend that time:

1. **Deciding whether to keep the recorded demonstrations.**
   `logs/demos/lift_sparse_qualify/demos.hdf5` holds ten valid demonstrations
   recorded with this code at seed 42, and it is usable as-is. Re-record if the
   study wants a different seed or a fresh wave.
2. **The offline BC fit at the full 10,000 steps**, once per seed (42, 43, 44).
   The smoke fit above was 1,000 steps and is not the study's actor.
3. **The three online arms**, three seeds, 800,000 physical training
   interactions each, plus the `--set irl.gradient_steps=64` follow-up.
4. **External checkpoint evaluation** and the W&B comparison.

## Design decisions worth knowing about

- **Recording reuses the training collector.** `src/training/record.py` drives
  `EpisodeCollector`, so demonstrations and collected transitions are produced
  by the same code: same pre-reset successors, same two ending flags, same
  complete final transition. The recorder keeps only complete successful
  episodes and reports what the discarded attempts cost.
- **The teacher's observation is cut, not the student's.** The student sees an
  episode clock the PPO teacher was never trained on. The recorder reads the
  per-term observation widths off the observation manager and hands the teacher
  the columns it knows, rather than assuming where the extra ones ended up.
- **One feature vector, two bases.** Replay and the critics carry all seven
  tapped terms; the reward learner fits the six dense ones. `FeatureColumns` is
  the single place that translation happens, and it is applied to the IRL
  update, the validation feature means and their labels alike.
- **Fixed and learned coefficients add on one named basis.**
  `AdditiveLinearRewardScorer` is what makes the three arms differ by one
  configuration line: the sparse coefficient is one on the `environment_plus_learned`
  arm and zero on the `learned` arm, and is never learned on either.
- **A demonstration file names the episode it was recorded on.** Horizon, step
  duration and the finite-horizon flag are stored and checked against the run
  that loads it.
- **A feature request is a subset request.** One recording serves arms that
  learn over different subsets of it; a missing requested column is still an
  error.

## Open questions for the experiment

- `runner.transition_budget` counts retained training transitions, which equals
  physical training interactions here only because every collected episode runs
  to the horizon. A run with many early failures will spend fewer physical
  steps than the budget suggests; check `Summary/environment_transitions`.
- The 64-reward-step follow-up is a CLI override (`--set irl.gradient_steps=64`)
  rather than a fourth config file, to keep the arms differing by one line.
