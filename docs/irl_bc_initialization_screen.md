# Pure IRL + BC: actor initialization and online BC weight

This screen isolates two choices that the earlier [mixed-reward weight screen](ten_demo_weight_ablation.md) did not: pure learned reward, and whether the actor is initialized by offline BC. The task's sparse terminal-success event is **only an evaluation metric** here; it is not part of the training reward.

For every arm, the reward learner fits a linear reward from the same ten complete successful demonstrations and the most recent batch of 32 complete attempted episodes. The actor maximizes `dt * dot(w, phi)` through successor PMD while retaining an action-MSE BC term:

```
L_actor = (1 - b) * L_PMD[dt * dot(w, phi)] + b * L_BC-MSE
```

Here `b` is `bc.weight`; it is a loss coefficient, **not** the measured fraction of the actor gradient attributable to BC. Test `b = 0.50, 0.90, 0.99`, equivalent to nominal PMD coefficients `0.50, 0.10, 0.01`. For each weight, compare:

- **BC actor initialization:** load the fully fitted, spread-calibrated ten-demo checkpoint into both the actor and PMD reference policy.
- **Random actor initialization:** load no actor checkpoint. The actor architecture and configured initial latent standard deviation remain identical, but its actual parameters and initial action distribution differ from the BC checkpoint. This arm still uses demonstrations for online BC, protected replay, the reward learner, and value warm-up; it is **not** a no-demonstration arm.

All six arms use seed 42, the same ten-demo file, Easy Lift with cube XY half-range 0.06 m, no curriculum, finite horizon 250, gamma 1, and terminal success defined by the final 25-step hold. Each collects 32 complete episodes per batch, retains short failures, gives replay 50% protected-demo sampling, and uses UTD 2 with one PMD improvement per batch. The actor learning rate is `3e-6`, the maximum projection length is eight steps, and both successor and separate entropy critics receive 10,000 demo-only optimizer steps before interaction. Reward learning takes eight steps per batch. The nominal online budget is 40,000 retained transitions, with a hard 48,000 physical-interaction cap; if failures shorten episodes, batch/trajectory and physical-interaction counts need not coincide. Run at most two Isaac processes at once on the single GPU.

The BC actor is `logs/benchmarks/lift_xy06_mse_spread_20260923/calibrated/floor_0.005.pt`; the ten demonstrations are `logs/benchmarks/lift_xy06_bc_scratch_20260923/demos_10.hdf5`. The base experiment is `configs/lift_cube_franka/irl_bc.yaml`. Set `env.lift_cube_xy_half_range=0.06`, `algo.pmd.policy.std_parameterization=smooth_bounded`, `algo.pmd.policy.minimum_std=0.005`, `algo.pmd.train.initial_critic_updates=10000`, `algo.pmd.actor_learning_rate=3e-6`, `algo.pmd.policy_improvements_per_batch=1`, `runner.save_interval=1`, `runner.transition_budget=40000`, `runner.total_transition_budget=48000`, and the common demonstration replay path. Set `--init_policy_checkpoint` only for the BC-initialized arms.

Score saved checkpoints in a **separate** Isaac process, using identical reset seeds and complete attempted episodes for each actor and both deterministic and stochastic policy modes. Evaluation transitions never enter replay. Compare held-goal terminal success primarily against physical training interactions; report feature gap with its collection-policy version, actual reward/critic/actor optimizer counts, collected trajectories, and evaluation cost separately. A single training seed is only a screen: any apparent winner needs fresh training seeds and a larger independent evaluation before claiming improvement.
