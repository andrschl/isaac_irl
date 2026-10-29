# BC demonstration-count screen — 2026-09-22

Reduce demonstration count, not fitting duration. This is a fixed-recipe
screen, not a claim that BC has been optimally tuned for each data budget.

## Controlled choices

- Actor seed 42, unchanged `(256,128,64)` ELU architecture, action bound 4,
  latent standard deviation 0.1.
- Every actor completes 40,000 Adam updates: MSE, batch 256, cosine learning
  rate from 3e-4 to 3e-6, no early checkpoint selection.
- L2 stays at zero, as selected by the preceding ten-demo validation study.
  This does **not** establish that zero is optimal at lower demo counts.
- Same sparse finite-horizon Easy Lift task, horizon 250, final-25-step hold
  criterion, failure precedence and no curriculum.
- Source: `logs/demos/lift_sparse_qualify/demos.hdf5`. Sorted demo indices are
  shuffled with a local PyTorch generator seeded 42, giving
  `[2,6,1,8,4,5,0,9,3,7]`. Two demos use `[2,6]`; three use `[2,6,1]`;
  five use `[2,6,1,8,4]`.
  Subsets were fixed before evaluation, not chosen for their success rates.
- The ten-demo actor is reused unchanged from the preceding full-fit study.
  Each actor is evaluated in a fresh Isaac process on reset seeds 6001/6002,
  32 attempted trajectories per seed, deterministic actions only. Common
  seeds do not guarantee bitwise-identical simulator trajectories.

Unused demonstrations are evaluated for action-error diagnostics **after**
fitting; they never supply optimizer batches or select a checkpoint/L2 here.
The inherited hyperparameters were selected with ten-demo validation, so this
is not an independent claim that only two/five expert episodes were available
for all method development. Smaller data budgets may require retuning.

## Reproduce the smaller fits

Run for `COUNT=2`, `COUNT=3` and `COUNT=5`, using new output paths:

```bash
python scripts/train_bc.py --task Isaac-Lift-Cube-Franka-v0 \
  --config configs/lift_cube_franka/sparse_irl_bc.yaml --seed 42 \
  --expert_data_path logs/demos/lift_sparse_qualify/demos.hdf5 \
  --expert_num_trajectories "$COUNT" --set expert_data.subset_strategy=random \
  --gradient_steps 40000 --learning_rate 3e-4 --final_learning_rate 3e-6 \
  --l2_weight 0 --batch_size 256 --log_interval 2000 \
  --output "logs/bc/new_bc_${COUNT}_demos_s42.pt"
```

Artifacts and W&B-linked fit metadata:
`logs/benchmarks/lift_bc_demo_count_20260922/`.
No source-code behavior or experiment defaults changed for this screen.

## Results

| Training demos | Expert transitions | Successes / 64 | Deterministic success | Seeds 6001 / 6002 |
| --- | ---: | ---: | ---: | --- |
| 2 | 500 | 0 | 0% | 0/32, 0/32 |
| 3 | 750 | 50 | 78.125% | 27/32, 23/32 |
| 5 | 1,250 | 61 | 95.3125% | 29/32, 32/32 |
| 10 | 2,500 | 49 | 76.5625% | 25/32, 24/32 |

The initial comparison was 2/5/10; three demos were added as an intermediate
count after seeing two fail and five succeed. Episode identities still came
from the original fixed ordering. No count reached the previous 30–60% target;
no premature checkpoint was selected to manufacture it.

Three demos are a promising reduced-data baseline, not a robust optimum.
This is one actor seed and one nested subset ordering. Five outperforming ten
here does not establish that fewer demonstrations generally improve BC.
Replicate subsets/training seeds before making that claim or choosing the
full RL/IRL study's expert budget. The ten-demo actor previously scored 52/64
on different reset seeds; the table uses the common **new** seeds instead.

Training MSE / unused-demo MSE was 3.6744e-6 / 4.9750e-4 for two demos,
7.9330e-6 / 4.0645e-4 for three, and 1.5052e-5 / 7.4564e-5 for five.
These are different unused episode sets, so their MSEs are not a common-test-set
ranking. The large small-data generalization gaps also preclude claiming
overfitting has been eliminated. Low supervised error and closed-loop success
measure different things.

- [Two-demo evaluation](https://wandb.ai/andrschl/isaac_irl/runs/wqpydfsj)
- [Three-demo evaluation](https://wandb.ai/andrschl/isaac_irl/runs/6eaaql3u)
- [Five-demo evaluation](https://wandb.ai/andrschl/isaac_irl/runs/gz7docsn)
- [Ten-demo control](https://wandb.ai/andrschl/isaac_irl/runs/84cardxx)

The candidate actor is `bc_3_demos_s42.pt` under the artifact directory above;
`summary.json` records counts, costs and paths. This screen spent 120,000 new
offline optimizer steps, **zero online training interactions**, and 64,000
evaluation interactions over 256 attempted full trajectories, with no discarded
transitions. Evaluation data was never optimized; no new demos were recorded.

For a subsequent reduced-demo RL/IRL comparison, use the **same exact subset**
for initial BC, online BC, IRL expert expectations, and protected demo replay.
Do not silently give the online arms the remaining expert episodes. Verify
subset propagation before launching those arms; no online runs were started
by this screen. Replicate demo-subset and actor seeds before treating a count
as a robust benchmark. Stochastic collection is still a separate qualification.

Current replay loading has its own `demonstration_episodes` cap and takes the
first episodes; it does not share BC's random subset selection. Merely setting
`expert_data.num_trajectories=3` is therefore insufficient for the online study.
Use a file containing exactly the chosen episodes for every consumer, or wire
shared episode identities through replay loading before that study starts.
