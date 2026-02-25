# Config Layout

- `configs/train.yaml`: canonical base IRL training config.
- `configs/train/<task_slug>.yaml`: thin task override config.
- `configs/env/*.yaml`: environment presets.
- `configs/policy/*.yaml`: policy architecture settings.
- `configs/algo/*.yaml`: PPO algorithm settings.
- `configs/reward/dense_mlp.yaml`: reward model settings (canonical dense/linear feature reward).

## Hard-Break Changes

The train parser now rejects stale keys:
- `runner.reward_update_interval`
- `runner.imitator_buffer.store_discounted_feature_returns`
- `runner.expert_buffer.store_discounted_feature_returns`

Reward config keys must match `RewardModelCfg` directly:
- `reward.hidden_dims` (not `reward_hidden_dims`)
- `reward.is_linear` (not `reward_is_linear`)

Reward types currently supported by `scripts/irl/train_irl.py`:
- `dense_mlp`
- `dense`

IRL return settings (`irl`):
- `discount_gamma`: optional discount override for IRL reward updates (`null` = use PPO `algo.gamma`).
- `normalize_returns_by_episode_length`: divide discounted returns by episode length (default `true`).

## Task Slug Mapping

- `Isaac-Lift-Cube-Franka-v0` -> `lift_cube_franka` -> `configs/train/lift_cube_franka.yaml`
