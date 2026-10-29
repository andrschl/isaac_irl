# Longer PMD runs on one shared GPU

`scripts/tune_pmd_parallel.py` takes the two finalists from a completed
[screening sweep](sequential_tuning.md) and runs a seed-matched pair concurrently.
Concurrency changes experiment scheduling only: each learner owns an independent
Isaac process, replay buffer, RNG state, output directory and W&B run.

The September 20 follow-up compares quantile 0.50, UTD 2, and **16 versus 32
policy improvements per episode batch**. Each run has a 3.2-million physical
training-interaction cap: normally 400 batches of 32 episode attempts, or 12,800
trajectories. Native early failures can change the retained-transition count;
plots always use measured physical interactions. Reward, entropy rate 0.006,
separate entropy critic, gamma 0.98, easy resets, curriculum-off setting, networks,
and replay capacity are unchanged. Once replay fills, its normal oldest-data
eviction still applies.

## Scheduling and evaluation

- Two concurrent learners, with simulator initialization staggered so both apps
  do not initialize shared caches together.
- Train continuously to the interaction cap; do not restart from a checkpoint
  that lacks replay. Every longer run starts from scratch.
- Wait for both training processes to exit before evaluating saved checkpoints,
  one simulator at a time. Training RNGs and environments are never reset by
  validation.
- Evaluate near 80k, 200k, 400k, 800k, 1.6M, 2.4M and 3.2M interactions, followed
  by a final standard-reset transfer check. Each point uses 128 evaluation
  trajectories: two seeds, 32 environments, deterministic and stochastic modes.
- Repeat the pair on seeds 42, 43 and 44, in that order, only while time permits.
  The first pair measures actual concurrent throughput. Later pairs need a
  10% time margin over that measurement, including evaluation and publication.

The controller holds the same exclusive *controller* lock as the sequential
sweep. The lock no longer means there is only one training process: the parallel
controller explicitly owns at most two. Unrelated GPU processes block launch;
the controller never kills them. Source hashes are checked before each launch.

```bash
python scripts/tune_pmd_parallel.py logs/benchmarks/<new-experiment> \
  --reference-status logs/benchmarks/<completed-screening>/status.json \
  --training-budget 3200000 --max-hours 24 --seeds 42 43 44
```

`--prepare-only` validates configs and records the protocol without launching.
`status.json` lists active process IDs and trial states. `gpu_usage.jsonl` samples
GPU utilization, memory and power every 30 seconds during training. Throughput
must be judged alongside utilization: a busier GPU alone does not prove faster
experiments, and shared-GPU wall times are not isolated algorithm speed results.

The 24-hour follow-up budget begins when the new controller starts. It includes
training, evaluation and dashboard publication. A `STOP` file in the experiment
directory or SIGTERM requests shutdown of owned processes. The wall-clock
watchdog does the same, preserving checkpoints and marking unfinished trials
incomplete rather than interpreting them as zero success.

W&B keeps the previous PPO run as a labeled reference. New evaluation curves
appear after each pair finishes training; live learning metrics are available
immediately in the individual training runs. The report overlays identical
metric keys against physical interactions, including failure rates and separate
evaluation-cost axes. Repeated seeds are separate curves, not retuned settings.
