# BC ensemble as trajectory coverage on Lift

This is an **evaluation of policy coverage**, not action averaging. The five
fully fitted ten-demo BC policies remain distinct. For each matched initial
state, run independent stochastic full trajectories under two conditions:

1. **Single BC:** sample actions repeatedly from the same calibrated BC
   policy.
2. **BC ensemble:** before each trajectory, sample one of the five BC members
   uniformly with replacement, then keep that member for the whole episode.

No member is chosen using rewards, Q values, or hindsight. No actions or
weights are averaged. The earlier averaged-action screen answered a different
question and is not used here.

For the first `n` trajectories per initial state, report both:

- **Trial success fraction:** successful trajectories divided by all `n`
  attempts.
- **At-least-one success fraction:** fraction of initial states for which at
  least one of the `n` attempts succeeded.

The latter is an empirical best-of-`n` *coverage* measure, not the success
rate of a policy that can predict which attempt will succeed before acting.
Here `n = 1, 2, 4, 8`. Each prefix uses the actual sampled trajectories,
not an independence formula such as `1-(1-p)^n`.

## Matched evaluation protocol

The BC members are the seed-42–46 checkpoints in
`logs/benchmarks/lift_bc_ensemble_20260925/`. Each was fitted on all ten
demonstrations for 40,000 MSE steps. The single-policy control is the
previously selected calibrated BC checkpoint
`logs/benchmarks/lift_xy06_mse_spread_20260923/calibrated/floor_0.005.pt`.
The members' constant latent std (`0.0051`) is close to, but not identical
to, that control's fitted conditional spread; this is a comparison with the
actual incumbent BC, not a claim that only member diversity differs.

Use Easy Lift with cube XY half-range 0.06 m, no curriculum, horizon 250,
and terminal success defined by the final 25-step hold. Evaluate 32 states
from each of two reset seeds (8801/8802), with eight trajectories per state
per condition. Before each attempt, reseed and reset Isaac. The evaluator
compares the initial policy observations to the first attempt for that seed
and aborts if any component differs by more than `1e-5`. Thus a reported
coverage difference cannot be explained by comparing different starts.

The full comparison costs `64 × 8 × 2 × 250 = 256,000` physical evaluation
interactions and 1,024 attempted trajectories; it uses **zero** new online
training interactions and writes nothing to replay. The two-state/two-trial
smoke test adds 2,000 interactions and eight attempted trajectories.

The command is:

```bash
python scripts/evaluate_checkpoints.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_bc.yaml \
  --checkpoints logs/benchmarks/lift_bc_ensemble_20260925/member_s{42,43,44,45,46}.pt \
  --bc_coverage \
  --baseline_checkpoint logs/benchmarks/lift_xy06_mse_spread_20260923/calibrated/floor_0.005.pt \
  --coverage_trials 8 --episodes_per_seed 32 --num_envs 32 \
  --evaluation_seeds 8801 8802 --modes stochastic --logger noop --headless \
  --output_dir logs/benchmarks/lift_bc_coverage_20260929/coverage_8801_8802 \
  --set env.lift_cube_xy_half_range=0.06 \
  --set algo.pmd.policy.std_parameterization=smooth_bounded \
  --set algo.pmd.policy.minimum_std=0.005 \
  --set algo.pmd.policy.initial_std=0.0051
```

The per-state member choices, per-trial successes and returns, matched-start
checks, and aggregate curves are saved in `bc_coverage.json` under the output
directory. No W&B upload was made for this run.

## Results (2026-09-29)

The full eight-trial run completed. All 32 attempts per condition/seed/trial
had **exactly matching initial policy observations** (`max difference = 0`).
The initial 64 states are shared between conditions, and each condition made
512 attempted trajectories. `Trial successes` is successes among the first
`64 × n` trajectories; `states covered` is the number of initial states with
at least one success in the first `n` attempts.

| Attempts per state `n` | Single BC: trial successes | Ensemble: trial successes | Single BC: states covered | Ensemble: states covered |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32/64 | 27/64 | 32/64 | 27/64 |
| 2 | 59/128 | 50/128 | 34/64 | 31/64 |
| 4 | 119/256 | 95/256 | 39/64 | 34/64 |
| 8 | **237/512** | **192/512** | **43/64** | **37/64** |

At eight attempts, 36 states were solved by both conditions, seven only by
the single BC, one only by the sampled ensemble, and 20 by neither. Thus
member sampling did expose one extra solvable state, but lost coverage on
seven others; it did **not** improve overall success probability or best-of-N
coverage in this test. The ensemble's trial-success fraction was 37.5%
versus 46.3% for the single BC, and its eight-attempt coverage was 57.8%
versus 67.2%. The five members were sampled approximately uniformly (draw
counts 97, 98, 97, 110, 110); one previously identified weak member is
included, not silently removed after evaluation.

This is a finite 64-state simulator estimate, not proof that a different
member set could never help. Crucially, it measures what was requested:
**different full BC policies sampled per trajectory versus repeated samples
from one fixed BC policy**. No averaged-policy behavior or online RL update
is involved. The raw results are
`logs/benchmarks/lift_bc_coverage_20260929/coverage_8801_8802/bc_coverage.json`.

## Without-replacement follow-up (2026-10-27)

The evaluator now also accepts `--coverage_member_sampling
without_replacement`. For **each initial state**, it shuffles all five BC
members and tries each exactly once in trials 1–5. Trials 6–8 use three
distinct members from a fresh shuffle. A chosen member remains fixed for
its entire episode. The original with-replacement mode remains available
and is the default.

The follow-up used the same five members, single-BC control, environment
configuration, reset seeds, 64 initial states, and eight stochastic trials
per condition. In the command above, change only the output directory to
`logs/benchmarks/lift_bc_coverage_20261027/without_replacement_8801_8802`
and add `--coverage_member_sampling without_replacement`. This additional
evaluation cost 256,000 physical interactions (1,024 attempted episodes),
with zero training interactions or replay insertions. Initial policy
observations matched exactly within each run (`max difference = 0`).

| Attempts per state | Single BC: successful trials | Without-replacement ensemble: successful trials | Single BC: states covered | Ensemble: states covered |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31/64 | 22/64 | 31/64 | 22/64 |
| 2 | 56/128 | 45/128 | 33/64 | 33/64 |
| 4 | 116/256 | 94/256 | 37/64 | 35/64 |
| 5 (all members tried) | 147/320 | 116/320 | 39/64 | 35/64 |
| 8 | **236/512** | **190/512** | **43/64** | **38/64** |

At eight attempts, both conditions covered 36 starts, only the single BC
covered seven, only the ensemble covered two, and neither covered 19. The
ensemble reached 38/64 starts, compared with 37/64 in the earlier
with-replacement run; this one-state difference is too small to establish
an improvement. The baseline had 9/512 trial outcomes differ from the
previous run despite matching reset/action seeds, while its final coverage
remained 43/64. Thus the two ensemble sampling modes were **not** directly
paired within a single simulator run, and the one-state difference should
not be interpreted as a controlled effect.

The weak seed-45 member still accounts for one of every first five
attempts per state; it succeeded in only 7/107 of its attempts in this run.
Without-replacement sampling guarantees member *diversity*, not success
coverage: members can fail on the same starts, and replacing a stronger
policy's attempt with a weak member can lower the chance of success. The
result does not show an ensemble advantage over repeated sampling from the
incumbent BC. Per-state assignments and outcomes are in
`logs/benchmarks/lift_bc_coverage_20261027/without_replacement_8801_8802/bc_coverage.json`.
