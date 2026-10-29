# Success-aligned Lift IRL screen

This is a matched follow-up to [the six-feature IRL screen](irl_bc_initialization_screen.md).
All online arms start from the same fully fitted ten-demo BC actor, use the same
ten demonstrations and protected replay, and collect five batches of 32 complete
attempted episodes. The initial BC evaluation is 32/64 deterministic and 33/64
stochastic successes on reset seeds 8501/8502. It costs zero **online training**
interactions; its demonstration and evaluation costs remain separate.

## Reward and units

Let the four bounded Lift task terms be `reaching_object`, `lifting_object`,
`object_goal_tracking`, and `object_goal_tracking_fine_grained`. Each lies in
`[0,1]`. With horizon 250 and `dt=0.02`, their episode integrals lie in `[0,5]`.
The learned basis divides each raw term by 5, so its episode integral is a
time average in `[0,1]`. The raw `terminal_success` term is 50 on a successful
final transition and zero elsewhere; its `dt`-weighted integral is already
either zero or one, so its divisor is 1. `action_rate` and `joint_vel` remain
in collection/replay/diagnostics, not the learned basis. No scale is estimated
from demonstrations or evaluation data.

For a frozen collection policy, exact linear IRL uses all ten expert episodes
and all 32 collected episodes to minimize

```
w · (mu_collection - mu_expert) + lambda * ||w||_2^2
subject to ||w||_2 <= 1,
mu = mean_episode[sum_t dt * raw_features_t / feature_scales].
```

The successor critic still learns raw feature returns; its scalarizer uses the
effective raw coefficient `w_i / feature_scale_i`. Thus reward learning and
policy control represent the same per-step reward. Exact IRL takes one analytic
solve and **zero reward-optimizer steps** per batch. Feature-gap metrics belong
to the policy that collected that batch, not the post-update policy.

| Arm | Reward used by PMD | Learned basis |
| --- | --- | --- |
| `task4` | scaled four-term reward | four task terms |
| `task4_success` | scaled four-term reward + learned success coefficient | four task terms + terminal success |
| `sparse` | fixed terminal +1 | none |
| `sparse_task4` | fixed terminal +1 and scaled four-term reward | four task terms |

The pure `task4_success` arm **does** use the environment's terminal success
event as a training feature. It must not be described as a reward-free
imitation arm. In `sparse_task4`, the +1 coefficient is fixed and the learner
cannot also move it.

## Matched launch

Use the `isaac_irl` Python environment. The ten-demo file and calibrated BC
actor below are the existing, qualified artifacts. Run the four commands with
the same common arguments, changing only source and learned basis. A failed
first batch is diagnosed in that run; do not create separate interaction-free
"smoke" runs and then omit their cost.

```bash
IRL_PY=/home/andreas/miniconda3/envs/isaac_irl/bin/python
IRL_DEMOS=logs/benchmarks/lift_xy06_bc_scratch_20260923/demos_10.hdf5
IRL_BC=logs/benchmarks/lift_xy06_mse_spread_20260923/calibrated/floor_0.005.pt
IRL_COMMON=(
  --task Isaac-Lift-Cube-Franka-v0 --headless
  --config configs/lift_cube_franka/irl_bc.yaml
  --expert_data_path "$IRL_DEMOS" --init_policy_checkpoint "$IRL_BC"
  --logger wandb --experiment_name lift_success_aligned_screen_20260924
  --max_iterations 5 --bc_weight 0.99
  --set "algo.pmd.train.replay.demonstration_path=$IRL_DEMOS"
  --set env.lift_cube_xy_half_range=0.06
  --set algo.pmd.policy.std_parameterization=smooth_bounded
  --set algo.pmd.policy.minimum_std=0.005
  --set algo.pmd.train.initial_critic_updates=10000
  --set algo.pmd.actor_learning_rate=3e-6
  --set algo.pmd.policy_improvements_per_batch=1
  --set runner.save_interval=1
  --set runner.transition_budget=40000
  --set runner.total_transition_budget=48000
)
IRL_EXACT=(
  --set irl.solver=exact_linear --set irl.gradient_steps=0
  --set irl.weight_decay=0
)
IRL_TASK4=(
  --set reward.learned_feature_scales.reaching_object=5
  --set reward.learned_feature_scales.lifting_object=5
  --set reward.learned_feature_scales.object_goal_tracking=5
  --set reward.learned_feature_scales.object_goal_tracking_fine_grained=5
)

"$IRL_PY" scripts/train.py "${IRL_COMMON[@]}" --run_name task4 \
  --reward_source learned "${IRL_EXACT[@]}" "${IRL_TASK4[@]}"
"$IRL_PY" scripts/train.py "${IRL_COMMON[@]}" --run_name task4_success \
  --reward_source learned "${IRL_EXACT[@]}" "${IRL_TASK4[@]}" \
  --set reward.learned_feature_scales.terminal_success=1
"$IRL_PY" scripts/train.py "${IRL_COMMON[@]}" --run_name sparse \
  --reward_source environment
"$IRL_PY" scripts/train.py "${IRL_COMMON[@]}" --run_name sparse_task4 \
  --reward_source environment_plus_learned "${IRL_EXACT[@]}" "${IRL_TASK4[@]}"
```

The saved `params/experiment.yaml` is an audit snapshot, **not** a CLI config:
evaluation must use the same base config and reward/policy overrides used for
training. Keep in-run validation disabled: it would use and reset the training
simulator. Instead, save `model_1.pt` through `model_5.pt` after their respective
policy updates (`runner.save_interval=1` above), then run
`scripts/evaluate_checkpoints.py` in a separate Isaac process with all five
checkpoints, in order, and a fresh `--output_dir`. Use
`--episodes_per_seed 32 --evaluation_seeds 8501 8502
--modes deterministic stochastic --logger wandb --headless`, plus:

```bash
--task Isaac-Lift-Cube-Franka-v0
--config configs/lift_cube_franka/irl_bc.yaml
--expert_data_path "$IRL_DEMOS"
--set experiment_name=lift_success_aligned_screen_20260924
--set env.lift_cube_xy_half_range=0.06
--set algo.pmd.policy.std_parameterization=smooth_bounded
--set algo.pmd.policy.minimum_std=0.005
--set reward.source=<this arm's source>
```

For learned arms, also pass the identical `IRL_TASK4` feature-scale overrides;
for `task4_success`, add its terminal-success scale of 1. The evaluator does
not build a reward learner, so it needs no solver override. It loads the
actor, successor critic and saved reward weights without inserting evaluation
transitions into replay.

One evaluator call per arm publishes **five points in one W&B run**, at 8k,
16k, 24k, 32k, and 40k physical training interactions. Every point evaluates
the *post-update* policy from that batch on the same 64 deterministic and 64
stochastic attempted episodes. Evaluation can be done after training because
the saved checkpoints are immutable and evaluation has no training feedback.
This costs at most 32k evaluation interactions per checkpoint; keep that cost
separate from the sample-efficiency x-axis. The standalone evaluator has no
`--experiment_name` option; set the W&B group through `--set` as shown.

The maximum initial screen is 160,000 online training interactions (640
attempted training episodes if none ends early), plus approximately 128,000
independent evaluation interactions. Log physical training interactions on
the x-axis and evaluation interactions separately. Advance an arm to training
seeds 43/44 only if final stochastic success reaches at least 38/64 and
deterministic success stays at least 32/64. This is a screening threshold,
not a significance claim. If none passes, inspect successor action ranking
and realized actor KL before increasing the interaction budget.

## Seed-42 screen result (24 September 2026)

All four runs completed five batches: **40,000 physical training interactions
and 160 complete attempted episodes per arm**. The four-arm total was 160,000
training interactions and 640 episodes. Independent final evaluation used 64
deterministic plus 64 stochastic attempts per arm, or 128,000 evaluation
interactions and 512 episodes in total; none entered replay. The BC model-0
reference was 32/64 deterministic and 33/64 stochastic on the same reset seeds.

| Arm | Deterministic | Stochastic | Train / evaluation W&B | Final checkpoint |
| --- | ---: | ---: | --- | --- |
| `task4` | 32/64 | 33/64 | [train](https://wandb.ai/andrschl/isaac_irl/runs/xlsbmf29) / [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/85h4njc6) | `2026-09-24_15-02-39_task4_s42/model_5.pt` |
| `task4_success` | 32/64 | 33/64 | [train](https://wandb.ai/andrschl/isaac_irl/runs/ue1r3si1) / [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/3atodim5) | `2026-09-24_14-51-26_task4_success_s42/model_5.pt` |
| `sparse` | 32/64 | 33/64 | [train](https://wandb.ai/andrschl/isaac_irl/runs/lpqx1mrv) / [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/x7uvlqoq) | `2026-09-24_14-54-27_sparse_s42/model_5.pt` |
| `sparse_task4` | 32/64 | 32/64 | [train](https://wandb.ai/andrschl/isaac_irl/runs/semafs8i) / [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/qn7xykb2) | `2026-09-24_15-07-06_sparse_task4_s42/model_5.pt` |

Checkpoints are under `logs/train/lift_success_aligned_screen_20260924/`;
machine-readable evaluations are under
`logs/benchmarks/lift_success_aligned_screen_20260924/<arm>_eval/model_5.json`.
Each critic received 10,000 demo-only warm-up steps and 80,000 online steps;
the separate entropy critic received the same. Each arm made five PMD policy
improvements. Actor Adam steps were 24, 13, 37, and 29 in table order. Each
learned-reward arm made five exact solves and zero reward-optimizer steps; the
sparse arm constructed no reward learner and made zero reward updates.

**No arm passed the predeclared 38/64 stochastic, 32/64 deterministic gate,**
so seeds 43/44 were not started. This screen provides no evidence of a
sample-efficiency improvement over BC. Last-batch collection-policy KL was
about 0.015–0.019: the policy changed modestly rather than remaining frozen.
Saved value diagnostics show reward-Q bias on final evaluation, but the saved
critic was fitted to the policy *before* the final actor projection; it is not
a same-policy Q calibration test and does not establish the cause. The
checkpoint evaluator does not retain state/action samples for an action-ranking
test, so that question remains unresolved. Do not infer a successful IRL
improvement from the feature gap alone: it measures the collection policy,
not the actor after the update.

## Independent evaluation after every batch

The above protocol was run for all `model_1.pt` through `model_5.pt` on reset
seeds 8501/8502. Entries are successes out of 64 attempts; each cell is an
independent checkpoint evaluation, **not** training-batch success.

| Arm | Deterministic, batches 1–5 | Stochastic, batches 1–5 | Five-point W&B run |
| --- | --- | --- | --- |
| `task4` | 32, 32, 32, 32, 32 | 34, 33, 33, 33, 33 | [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/l0pzue5w) |
| `task4_success` | 32, 32, 32, 32, 32 | 33, 33, 34, 33, 33 | [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/sjrv3c30) |
| `sparse` | 32, 32, 32, 32, 32 | 33, 33, 32, 33, 33 | [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/9kv3m4pk) |
| `sparse_task4` | 32, 32, 32, 32, 31 | 32, 32, 33, 33, 32 | [evaluation](https://wandb.ai/andrschl/isaac_irl/runs/1pnqdyqj) |

JSON for each point is under
`logs/benchmarks/lift_success_aligned_screen_20260924/<arm>_per_batch_eval/model_<batch>.json`.
The 20 checkpoint evaluations added 640,000 simulator interactions and 2,560
attempted evaluation episodes, **zero** online training interactions. The
original four final-checkpoint evaluations cost another 128,000 evaluation
interactions; they are separate runs, not additional training data. Repeating
`model_5.pt` reproduced the original final success counts exactly.

The training batches peaked together at batch 4 (24–25/32) then declined at
batch 5 (20–21/32), but these fixed-seed curves do not show a corresponding
policy-performance peak. The training metric is measured *before* that batch's
update: batch-4 collection used `model_3.pt`, whereas the 32k evaluation
point is `model_4.pt`. At either checkpoint, independent success stayed near
the BC baseline. This suggests that the shared training-batch swing largely
reflects which 32 episodes were sampled, though a smaller policy effect is
not ruled out by 64 evaluation attempts per mode.

The subsequent bounded actor/exploration tuning and its independent results
are in [the aggressive Lift tuning screen](aggressive_lift_tuning.md).
