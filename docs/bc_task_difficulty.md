# Harder BC benchmark through initial-state variation

The objective is a challenging **well-fitted** BC baseline with 10–100 complete
expert trajectories, ideally about 30–60% deterministic success. This is a
benchmark-design target, not a stopping rule for BC optimization. Never weaken
BC by choosing an early checkpoint, a poor teacher, or a test-only task change.

## Minimal task change

Keep `env.variant=lift_easy` for the pre-grasp robot pose and fixed goal. Set
`env.lift_cube_xy_half_range=0.06` to broaden only the cube's initial position:

| Property | Existing task | Broader candidate |
| --- | --- | --- |
| Cube x (m) | [0.48, 0.52] | [0.44, 0.56] |
| Cube y (m) | [-0.02, 0.02] | [-0.06, 0.06] |
| Spawn area | 0.0016 m² | 0.0144 m² (9×) |
| Goal | (0.5, 0, 0.2) | Unchanged |
| Robot start, action map and dynamics | Pre-grasp preset | Unchanged |
| Horizon / step duration | 250 / 0.02 s | Unchanged |
| Terminal success | Final 25-step hold, failure takes precedence | Unchanged |

The shared builder applies the range to recording, PPO/PMD training and
evaluation. Default width remains 0.02, so existing runs are unchanged. Values
must be finite and in (0, 0.1] m, staying within the native x workspace;
nondefault values require `lift_easy`. No robot joint noise, new observation
channels, reward changes or stricter success threshold were introduced.

## Qualification before BC

The original easy-task PPO expert scored **34/64 (53.125%)** on the broader
candidate at recording seed 142, below the existing 95% expert qualification
bar. This is **expert performance, not a BC result**. Ten successful episodes
were retained for diagnostics in `logs/demos/lift_xy06_qualify_20260922/`;
they must not be presented as a qualified broader-distribution expert dataset.
Selecting only successes from a teacher with this failure rate can strongly
bias the initial-state distribution of demonstrations.

The existing native-task PPO checkpoint
`lift_bench_20260919_moySrr/2026-09-19_10-00-22_ppo_s42/model_1500.pt` was also
checked on this same task/seed: **0/64**, so it is not an alternative qualified
teacher. Its empty diagnostic file is under
`logs/demos/lift_xy06_native_expert_qualify_20260922/`; no successful recording
report was produced because the one-wave attempt budget was exhausted.
Together the two qualifications cost 32,000 physical interactions and 128
attempted trajectories. No student-training interactions were collected.

The task change is implemented and verified in real Isaac. The initial screen
stopped before expert retraining; see the authorized continuation below.
The regression suite passes 464 tests, including unchanged
default-task behavior, isolated reset-range changes and bounded recording.
Publishing these two qualification results to W&B was initially blocked by the security
review because the proposed upload included local checkpoint paths and task
metadata. The user approved training and that export on 2026-09-23.

The recorder now accepts `--max_attempted_episodes`, checks the budget before
each full vector wave, and fails explicitly rather than collecting indefinitely.
Already written partial files are preserved. HDF5 attributes include the exact
environment configuration, recording seed and teacher checkpoint; successful
recordings also write `recording.json` with attempted/successful/saved counts
and total physical interaction cost. These provenance fields are descriptive;
the loaders do not yet automatically enforce initial-distribution equality.

For qualification, one saved demo can suffice: the recorder still scores the
entire 64-lane wave and separates failed attempts from surplus successes.

```bash
python scripts/record_demos.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --no-video --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --teacher_config configs/lift_cube_franka/ppo_easy.yaml \
  --checkpoint <qualified-teacher-checkpoint> \
  --set env.lift_cube_xy_half_range=0.06 --set seed=142 \
  --num_envs 64 --num_demos 1 --max_attempted_episodes 64 \
  --output_dir logs/demos/new_xy06_qualification
```

## Calibrating the eventual baseline

1. Obtain a PPO teacher qualified on the **same** broader reset distribution.
   If existing teachers fail, train/fine-tune PPO with the widened reset range;
   keep the final qualification rule fixed. Expert-training interaction costs
   are reported separately from student online interactions.
2. Record a new pool of 100 full trajectories on this distribution, retaining
   terminal observations and ending flags. Keep the old narrow-reset dataset
   intact and out of matched-distribution BC/IRL comparisons.
3. Screen fixed nested subsets of 10/30/100 with complete BC fits and
   trajectory-level validation of regularization. Inspect fitting stability;
   do not assume the old recipe is optimal on the new distribution.
4. Evaluate deterministically on separate reset seeds. Use development results
   to choose a task width, then freeze that width and confirm with new actor,
   subset and evaluation seeds. Report all candidates; do not keep searching
   on a supposed held-out test set. Stochastic PMD collection is a separate check.
5. If BC remains near ceiling at the intended demo budget, broaden the range
   incrementally (within the validated workspace) and repeat expert
   qualification and recording. Do not just broaden evaluation conditions.

The 30–60% band is not yet established. In particular, the existing expert's
53% score is not evidence that a properly taught BC student meets the goal.
No broader-task BC or student RL/IRL runs have been started yet.

## Authorized continuation — 2026-09-23

**Scope explicitly confirmed by the user: BC only. PPO is permitted solely to
obtain the demonstration expert. No student RL, PMD training or IRL is running
or queued.** The continuation consists of teacher training/qualification,
recording, offline BC fitting and independent evaluation only.

PPO expert fine-tuning started from the easy expert's actor. The critic and
optimizer are fresh; this is not a resumed run. Training uses the established
`ppo_easy.yaml` dense teacher objective (gamma 0.98, entropy rate 0.006,
continuing-timeout semantics), with the widened reset distribution. Teacher
qualification and all student experiments instead use the sparse finite-horizon
task and its final-25-step success criterion. The teacher's original 36-column
observation view is retained by the recorder; the student receives 38 columns.

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/ppo_easy.yaml --logger wandb \
  --experiment_name lift_xy06_expert_20260923 --run_name ppo_finetune_s42 \
  --seed 42 --num_envs 4096 --max_iterations 1500 \
  --init_policy_checkpoint logs/train/lift_easy_entropy_quantiles_20260919_BRLXuf/2026-09-19_17-49-28_ppo_soft_s42/model_1500.pt \
  --set env.lift_cube_xy_half_range=0.06 --set runner.save_interval=100
```

Budget: 1,500 × 4,096 × 24 = **147,456,000 expert-training interactions**,
separate from student online sample efficiency. Inline evaluation remains off.
Training output:
`logs/train/lift_xy06_expert_20260923/2026-09-23_08-32-02_ppo_finetune_s42/`.
[Expert-training W&B](https://wandb.ai/andrschl/isaac_irl/runs/mrtwqr3m).

The next stages are gated: qualify the completed teacher on fresh reset seeds;
if it reaches 95%, record a new 100-demo pool, create fixed nested 10/30/100
files preserving every transition/metadata field, run whole-episode BC
regularization validation and complete final fits, then evaluate on separate
reset seeds. Each online consumer can later read the same chosen subset file,
avoiding the current mismatch between BC random selection and replay's prefix
selection. No student RL/IRL sweep is authorized or launched by this continuation.

The warm-start attempt completed all 1,500 iterations (147,456,000 interactions,
589,831 episode endings including the initially staggered partial episodes), but
qualified at only **30/64 (46.875%)** on seed 242. Its BC pipeline stopped before
recording the new pool. [Qualification W&B](https://wandb.ai/andrschl/isaac_irl/runs/az09oh5z).
The inherited policy's mean exploration std started near 0.052 and ended near
0.038; insufficient exploration is a hypothesis, not an established cause.

A second bounded PPO expert attempt starts **from scratch**, otherwise using
the identical 1,500-iteration command (omit `--init_policy_checkpoint`, set
`--run_name ppo_scratch_s42`). Its planned additional cost is 147,456,000 expert
interactions, for 294,912,000 across both attempts if it completes. This is not
student RL training. Output is
`logs/train/lift_xy06_expert_20260923/2026-09-23_08-42-04_ppo_scratch_s42/`.
The BC-only gated continuation and live `state.json` are under
`logs/benchmarks/lift_xy06_bc_scratch_20260923/`; it reuses the first attempt's
workflow and again refuses to proceed unless the expert qualifies at 95%.
