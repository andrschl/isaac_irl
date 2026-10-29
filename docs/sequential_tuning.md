# Sequential Lift PMD tuning

One controller holds `logs/benchmarks/.isaac_gpu0.lock` for the sweep. It starts
one child, waits for exit, and checks that GPU 0 has no compute processes before
starting another simulator. No other job is killed to obtain the GPU. All
training, checkpoint evaluation, and dashboard publication run in sequence.

Training runs continuously with replay intact. After it exits, evaluate saved
checkpoints near 80k, 200k, 400k, 640k and 800k physical training interactions,
then evaluate the final policy on standard resets. Each evaluation uses 32
first episodes per seed, seeds 1001/1002, and deterministic/stochastic modes:
128 trajectories and at most 32k physical interactions. All evaluations,
including failures, must succeed for a trial to be selected.

## Experiment design

1. Screen quantiles 0.25/0.50/0.75 × UTD 1/2, with 32 policy improvements per
   batch and training seed 42: six trials.
2. Try 16 and 8 policy improvements per batch for the best screened
   quantile/UTD pair: two trials, same training seed and interaction budget.
3. Replicate the best two configurations from stages 1–2 on seeds 43 and 44,
   alternating configurations within each seed, if the remaining time permits.

Every trial starts from scratch and has an 800k **physical-interaction cap**,
normally 100 batches × 32 episodes = 3,200 trajectories. If every vector lane
terminates early, there can be more batches; the runner stops conservatively
before a whole batch could exceed the cap. Final counts can then be less than
800k by at most one 8k batch. The plots use measured counts, never nominal counts.

UTD counts successor-critic updates per retained transition; the separate
entropy critic is also updated on every critic step. The scalar entropy rate
remains 0.006 with dt=0.02, gamma=0.98. Reward, easy reset, curriculum-off setting,
actor and critic architectures, replay, Adam rates and action bounds are fixed.
PPO is the previous same-objective run, labeled as a reference; it is not retrained.

Selection averages the final two easy-task evaluations, weighting deterministic
and stochastic modes equally, and ranks lexicographically by sustained success,
ever reaching the goal, fewer early failures, then episode return. It does not
select by reward per step, which can increase when a policy terminates early.
This is a declared screening rule, not a significance test. Replications are
separate curves and are not used to retune the screened parameters.

## Launch and controls

```bash
python scripts/tune_pmd.py logs/benchmarks/<new-experiment> \
  --reference-status logs/benchmarks/<previous-comparison>/status.json \
  --evaluator logs/benchmarks/<previous-comparison>/evaluate.py \
  --max-hours 24
```

`--prepare-only` freezes the baseline/evaluator, validates six configs, and
writes the plan in `status.json` without launching anything. The same command
without this flag starts the prepared experiment, provided source hashes still
match. A used experiment directory cannot be restarted: the learner's replay is
not checkpointed. Existing trials and checkpoints are preserved.

The time budget includes training, evaluation, and publication. Conservative
estimates based on completed trials prevent launching a trial unlikely to fit;
skips are explicitly recorded. The watchdog stops the owned process at the
budget limit and marks the affected trial incomplete. Creating `STOP` in the
experiment directory requests the same stop. The controller never kills an
unrelated GPU process. Overlapping sweep controllers are prevented by the
exclusive lock and one-child-at-a-time contract.

Read `status.json` and `controller.log` for progress and `evaluation_dashboard.json`
for the shared W&B report. Legends specify q, UTD, P and seed. Every evaluation
chart uses physical interaction counts, and early-failure rates are shown next
to reward, sustained success and lifting. Evaluation costs are reported separately
and on a second total-interaction axis. Source/config changes abort the sweep
before the next child starts so results cannot mix implementations.
