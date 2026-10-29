# Comparing checkpoint evaluations

Use the **comparison report** linked in `evaluation_dashboard.json`. Every
panel overlays PPO and the successor-PMD quantiles, using the same metric key
and a physical environment-interaction x-axis. The controller's original
`checkpoint_eval` run is a diagnostic archive, not the comparison view.

Publish existing evaluations (no simulator needed):

```bash
python scripts/publish_evaluations.py logs/benchmarks/<experiment>/status.json \
  --entity andrschl --project isaac_irl --watch
```

The publisher backfills saved results and follows new ones until the controller
stops. It records W&B run ids, published checkpoints and the report URL in
`evaluation_dashboard.json`. Rerunning resumes those evaluation streams and
skips checkpoints already published. It never writes to training histories or
restarts a learner. W&B's `wandb-workspaces` package is needed for report creation.

## Axes and curves

- **Training environment interactions**: actual transitions across all vector
  lanes, including lane overhead. This is neither checkpoint number, replay
  size, nor the number of vectorized `env.step()` calls.
- **Including evaluation costs**: training interactions at that checkpoint
  plus cumulative external evaluation interactions, including transfer checks.
  Each variant retains the cost charged when it was evaluated.
- **Reward per step**: sum of task rewards divided by valid evaluation
  transitions. Entropy bonuses are excluded. Short episodes contribute their
  actual length, not padded zeros or a fixed-horizon denominator.
- **Sustained success**: fraction of attempted episodes within the goal
  tolerance for every one of the final 25 steps.
- **Lift rate**: fraction of episodes lifting the cube at least 10 cm.

The first section uses a linear x-axis limited to the latest evaluated budget
available for **every** learner. The full-range section uses a logarithmic
x-axis so PPO's larger budget does not visually squash PMD. Both use observed
checkpoint scores, without smoothing or extrapolation. Lines between points
are visual guides, not extra measurements. The deterministic and stochastic
policies occupy separate rows; standard-reset transfer has its own section.

Each evaluation stream has the same `Evaluation/<variant>/<mode>/*` keys and
binds them to `Samples/training_environment_interactions`. The report explicitly
sets its x-axis too, rather than inheriting W&B's internal logging counter.
Legends identify algorithm, quantile and training seed. Evaluation streams are
tagged `evaluation` and grouped separately from learner runs: they are copies
of saved measurements, not additional training experiments.

The current protocol uses two evaluation seeds, 32 episodes each per policy
mode, and only one training seed. Episode sampling noise and training-seed
variability are different quantities; the report makes no multi-seed claim.
