# Ten-demo BC spread calibration

The fresh NLL actors failed because their likelihood loss could widen spread
around poor action means. Nine teacher actions are exactly +4 on the first step
of nine of the ten demos: they were clipped at execution, while a tanh Gaussian
places no probability mass exactly on its ±4 boundary. Inverting them after a
numerical clamp creates artificial latent targets around 7.25. The smallest-floor
NLL actor predicted only 1.29 for those actions and assigned latent std 1.91;
the fully fitted MSE actor predicted 3.95.

## Fix

Keep the ten-demo MSE actor's trunk and mean head **bit-for-bit unchanged**.
The saved five MSE whole-episode fold checkpoints provide honest held-out means;
the all-ten checkpoint supplies the final mean. Copy those two modules into the
smooth-bounded actor, freeze them, and fit only the std head. The optimizer sees
only recorded actions strictly inside ±4, and reports excluded boundary pairs.
The MSE mean still trains on **all** actions, including the clipped ones.

The spread objective is mean negative log density of interior executed actions.
All five fold fits use the full 10,000-step Adam budget, batch 256, cosine
learning rate 1e-3 to 1e-5. Fit floors 0.001, 0.005 and 0.02 on the same ten
demonstrations. Then fit every floor on all ten; the file used by the online
arms is one of these complete fits, not an early fold checkpoint. Online BC
remains action MSE, so it cannot reintroduce the clipped-action NLL problem.

For bounded PMD policies, the ordinary BC NLL API now rejects a clipped expert
action with an explicit error. The PPO Gaussian BC path is unchanged.

## Selection and accounting

Five-fold held-out interior NLL ranks floor 0.02 first:

| Floor | Held-out interior NLL | Full-fit mean-action MSE | Full-fit mean latent std |
| --- | ---: | ---: | ---: |
| 0.001 | 9.12 | 0.0000699 | 0.00170 |
| 0.005 | −10.74 | 0.0000699 | 0.00509 |
| 0.02 | −13.44 | 0.0000699 | 0.02003 |

Because held-out density previously failed to rank task success, evaluate all
three complete actors on calibration reset seeds 8301/8302, in both policy
modes. Choose the highest stochastic success, with held-out NLL breaking ties.
Then report the chosen actor and original MSE reference on untouched reset
seeds 8401/8402. Evaluation episodes never enter fitting or replay, and every
arm has zero online training interactions at this stage.

On calibration seeds, every floor and the original MSE actor reached 40/64
deterministic successes, confirming that the mean was preserved. Stochastic
success was 37/64 at floors 0.001 and 0.005, 20/64 at 0.02, and 0/64 for the
original actor. The tie goes to floor **0.005**, with lower held-out NLL than
0.001. On untouched seeds 8401/8402, floor 0.005 reached **35/64 deterministic,
34/64 stochastic**; the original MSE actor reached **35/64 deterministic,
0/64 stochastic**. Thus the repair improves stochastic control while keeping
the tested deterministic mean behavior exactly.

The selected checkpoint is
`logs/benchmarks/lift_xy06_mse_spread_20260923/calibrated/floor_0.005.pt`.
Set the online BC regularizer to MSE and use the same ten-demo file for expert
features and protected replay. Evaluations each used 64 attempted episodes per
policy mode, counted separately from online training.

The command and output are:

```bash
python scripts/calibrate_bc_spread.py --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml \
  --expert_data_path logs/benchmarks/lift_xy06_bc_scratch_20260923/demos_10.hdf5 \
  --mean_checkpoint logs/benchmarks/lift_xy06_bc_scratch_20260923/xy06_bc_10d_s42.pt \
  --fold_checkpoint_dir logs/benchmarks/lift_xy06_bc_scratch_20260923/xy06_bc_10d_s42_validation \
  --std_floors 0.001 0.005 0.02 --gradient_steps 10000 \
  --set env.lift_cube_xy_half_range=0.06 \
  --output_dir logs/benchmarks/lift_xy06_mse_spread_20260923/calibrated
```

The calibration run is [on W&B](https://wandb.ai/andrschl/isaac_irl/runs/r88ecieg).
The selected actor's policy configuration must set
`algo.pmd.policy.std_parameterization=smooth_bounded` and its exact floor for
checkpoint loading. Its std bounds are saved in the actor state and checked at
load time. It can then be used for demo-only value warm-up and PMD fine-tuning.

## First PMD-batch check

Both checks loaded the selected actor into the live and reference policies,
performed 10,000 demo-only updates of each value component, then collected one
batch of 32 complete episodes (8,000 physical interactions) and performed 16
PMD improvements. They used the same ten demos, seed 42, half sparse / half
learned reward, and the same independent evaluation seeds 8501/8502. Only the
online BC-MSE weight differed. Success counts below are out of 64 attempted
episodes per mode; evaluation episodes did not enter training.

| Online BC weight | Checkpoint | Deterministic | Stochastic |
| --- | --- | ---: | ---: |
| 0.1 | warm-up, 0 interactions | 32 | 33 |
| 0.1 | first batch, 8,000 interactions | 19 | 26 |
| 0.9 | warm-up, 0 interactions | 32 | 33 |
| 0.9 | first batch, 8,000 interactions | 16 | 24 |

The mean-preserving spread fix solves the **BC initialization** failure, not
the subsequent PMD regression. Raising the raw MSE weight from 0.1 to 0.9 did
not prevent it; the MSE and PMD losses have very different numeric scales, so
the coefficient is not a percentage of their gradient influence. The warm
critic's start-state reward and entropy estimates also disagree substantially
with sampled returns on the evaluation rollouts. Those sampled returns are
noisy diagnostics, not exact conditional Q targets. Do not treat either
one-batch run as evidence of successful online fine-tuning.

The [0.1 training run](https://wandb.ai/andrschl/isaac_irl/runs/izt53aao)
and [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/rchsy2gh), and the
[0.9 training run](https://wandb.ai/andrschl/isaac_irl/runs/221c5ml5) and
[evaluation](https://wandb.ai/andrschl/isaac_irl/runs/p7vr9axx), preserve the
full diagnostics.
