# Config Layout

- `configs/franka_lift/experiment.yaml`: single experiment config (env, reward, policy, algo, irl, runner, bc, etc.).

## Reward Types

- `dense`: MLP reward model. Uses `hidden_dims`, `activation`. Default: L2 regularization.
- `linear`: linear reward model (r = w^T x). Uses `linear_projection`. Default: L2 ball projection.

Field applicability:
- Dense: `hidden_dims`, `activation` apply; `linear_projection` invalid when not linear.
- Linear: `linear_projection`, `linear_projection_radius` apply; `hidden_dims`, `activation` ignored.
- Both: `regularization`, `regularization_strength`, `elastic_alpha`.

## Hard-Break Changes

The train parser rejects stale keys:
- `runner.reward_update_interval`
- `runner.num_steps_per_env_rl`
- `runner.policy_updates_per_cycle`
- `runner.imitator_buffer.store_discounted_feature_returns`
- `runner.expert_buffer.store_discounted_feature_returns`

Reward config keys must match `RewardModelCfg` directly:
- `reward.hidden_dims` (not `reward_hidden_dims`)
- `reward.is_linear` (not `reward_is_linear`)

IRL return settings (`irl`):
- `discount_gamma`: optional discount override for IRL reward updates (`null` = use PPO `algo.gamma`).
- `normalize_returns_by_episode_length`: divide discounted returns by episode length (default `true`).

Runner update settings (`runner`):
- `use_learned_reward`: train the agent on the learned reward model instead of the raw env reward.
- `steps_per_env_per_cycle`: vectorized env steps collected before an agent update.
- `rl_updates_per_cycle`: RL-algorithm update calls per learning iteration.

Runner validation settings (`runner`):
- `validation_interval`: run stochastic + inference validation every N learning iterations (`0` disables).
- `validation_steps_per_env`: env steps per validation mode (`null` = `5 * steps_per_env_per_cycle`, `0` disables).
- Validation emits feature returns only: `ValidationFeatures/<mode>/{expert,imitator,gap}/<feature>`; old reward-update `EvalFeatures/*` logs are intentionally disabled.

## Policy algorithms

The public choices are `ppo`, `sac`, and `pmd`. SAC and PMD select either
`algo.value.type: direct` or `successor`; these are value representations,
not separate algorithms. See [the RL algorithm reference](../docs/rl_algorithms.md)
for the exact updates and parameter meanings.

PMD keys that decide sample efficiency: `critic_updates_per_projection` (how
many critic steps evaluate the frozen policy before each projection),
`actor_projection_kl` (hard trust region per projection) and
`candidate_target_kl` (the nonparametric target it aims at, which must be
larger). CLI equivalents exist for all three:
`--pmd_critic_updates_per_projection`, `--pmd_actor_projection_kl`,
`--pmd_candidate_target_kl`, alongside `--rollout_steps`.

## Behavioral cloning (`bc`)

BC is independent of the RL algorithm:

`L_actor = (1 - bc.weight) * L_RL + bc.weight * L_BC`.

The dataset is read from `irl.expert_data_path`. `weight=0` is the default
and constructs no BC loader, batch, or gradient.

Fields:

- `weight` (float in `[0,1]`): BC mixing weight.
- `loss_type` (`nll` or `mse`): expert action likelihood or mean regression.
- `batch_size`: expert minibatch size per actor optimizer step.
- `val_fraction`: fraction of whole expert episodes reserved for validation.

CLI overrides are `--bc_weight` and `--bc_loss_type`.
