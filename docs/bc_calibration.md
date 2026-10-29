# BC initialization calibration — 2026-09-22

**Superseded as a selection protocol:** the user subsequently requested a
properly fitted, regularized BC baseline, not a checkpoint chosen to land in a
30–60% success band. The results below remain historical measurements, not the
recommended selection rule. BC should be optimized for generalization even if
it exceeds 60% success.
See [Properly fitted BC](bc_regularization.md) for the replacement procedure.

## Qualified candidate

The seed-42 BC actor reached **30/64 deterministic successes (46.875%)** on
fresh reset seeds 4001 and 4002: 14/32 and 16/32 respectively. This meets the
requested 30–60% deterministic starting range. With only 64 trials it is an
estimate, not an exact success probability (95% Wilson interval about 35–59%).

- Actor: `logs/benchmarks/lift_bc_screen_20260922_74m4db/bc_baseline_s42.pt`.
- Evaluated weights: `bc_refine_1000.pt` in the same directory. The named
  baseline is reproduced via the normal CLI and checked for exact equality.
- Results: `confirmation_deterministic_4001_4002/bc_refine_1000.json`.
- [Confirmation on W&B](https://wandb.ai/andrschl/isaac_irl/runs/6om7s60j).

The task, termination rule, ten demonstrations, 38-input actor architecture,
action bounds, uniform-pair MSE and batch size 256 are unchanged. The change is
offline optimizer scheduling: 20,000 Adam steps at 3e-4, followed by 1,000 steps
at 3e-5 with a fresh Adam optimizer. Refinement loads actor weights, not optimizer
state. No RL, reward update, extra demonstration or evaluation-label training
is involved.

## Reproduction

Use new output paths to preserve existing experiment artifacts:

```bash
python scripts/train_bc.py --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml --seed 42 \
  --expert_data_path logs/demos/lift_sparse_qualify/demos.hdf5 \
  --gradient_steps 20000 --learning_rate 3e-4 --batch_size 256 \
  --output logs/bc/new_bc_base_s42.pt

python scripts/train_bc.py --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml --seed 42 \
  --expert_data_path logs/demos/lift_sparse_qualify/demos.hdf5 \
  --init_policy_checkpoint logs/bc/new_bc_base_s42.pt \
  --gradient_steps 1000 --learning_rate 3e-5 --batch_size 256 \
  --output logs/bc/new_bc_baseline_s42.pt

python scripts/evaluate_checkpoints.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --expert_data_path logs/demos/lift_sparse_qualify/demos.hdf5 \
  --checkpoints logs/bc/new_bc_baseline_s42.pt \
  --output_dir logs/evaluations/new_bc_confirmation \
  --num_envs 32 --episodes_per_seed 32 --evaluation_seeds 4001 4002 \
  --modes deterministic --logger wandb
```

Actor-only evaluation explicitly omits critic diagnostics: it never presents
newly initialized critics as trained values. Each BC candidate has its own
W&B evaluation stream at zero online training interactions. Missing stochastic
metrics mean that mode was not evaluated, not zero success.

## What was learned

Development used reset seed 2001 and 16 episodes per mode. Confirmations used
fresh reset seeds and 64 episodes per mode, in a new simulator process.

| Fit | Development deterministic | Independent deterministic confirmation |
| --- | ---: | ---: |
| Original 10k steps at 3e-4 | 0/16 | 0/64 (earlier pilot evaluation) |
| 20k steps at 3e-4 | 1/16 | Not evaluated |
| 20k + 500 refinement steps at 3e-5 | 9/16 | 44/64 (seeds 3001/3002) |
| **20k + 1,000 refinement steps at 3e-5** | **8/16** | **30/64 (seeds 4001/4002)** |
| 20k + 5,000 refinement steps at 3e-5 | 13/16 | Not evaluated |

All trials, including other durations, a lower-rate fit from scratch, shorter
refinements and secondary noise checks, remain under the artifact directory.
More fitting did not monotonically improve closed-loop success. The original
10k actor had full-demo MSE around 1e-4 and correctly aligned observations and
actions, yet failed in closed loop. Lower-rate refinement was decisive in this
screen; low training MSE alone was not a qualification criterion.

This is a deliberately moderate initial policy, **not a claim that BC cannot
do better with ten demos**. Retain the stronger 5,000-refinement-step candidate
as a reference and report the initialization selection rule in comparisons.
Use the same selected actor in every online arm. Repeat the fixed recipe for
training seeds 43/44 without choosing different per-seed budgets to force 50%.
Those replications have not been run.

Identical mean policies showed some variation in sequential in-process Isaac
screens despite reseeding. The cause is not established; common seeds should
not be treated as guaranteeing bitwise-identical contact trajectories. Use
separate processes for final confirmations and report multiple reset seeds.

## Collection noise is not yet qualified

The selected actor retains latent std 0.1 and the original minimum std 0.05.
Its stochastic development score was 0/16. The user selected deterministic BC
as the first milestone; a good deterministic policy is not yet a good PMD
collection distribution.

Secondary development checks on the **500-refinement-step actor**, not the
selected 1,000-step actor, reached 7/16 stochastic successes at latent std
0.005. These used explicit minimum std 0.005 and unchanged mean weights; no
evaluation-only temperature was used. This is a lead for collection calibration,
not a confirmed setting or a change to experiment defaults. A different mean
policy with 5,000 refinement steps reached 4/16 at std 0.01.

Before RL: qualify collection noise on the chosen actor, then fit fresh
successor/entropy critics for that exact stochastic policy. Do not reuse the
old pilot's critics or interpret 1,000 TD steps as proof of accurate values.

## Cost and scope

This calibration consumed **184,000 physical evaluation interactions / 736
attempted trajectories**, with 183,512 retained transitions and 488 discarded
vector-lane transitions. None entered optimization. This cost includes every
failed trial and secondary noise check, but excludes prior expert recording,
the 80k pilot and its earlier 64k evaluation. Online training cost for these
BC candidates is zero; their evaluation and selection cost is not zero.

No PMD/IRL experiment or cross-seed training replication was started here.
