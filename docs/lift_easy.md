# Easier Lift: reset and goal geometry only

`env.variant: lift_easy` is an opt-in preset for `Isaac-Lift-Cube-Franka-v0`.
`standard` remains the default; other tasks reject the Lift-only preset.

Cube reset width can be varied without changing the rest of this preset:
`--set env.lift_cube_xy_half_range=0.06` makes x uniform on [0.44, 0.56] and
y uniform on [-0.06, 0.06]. The default 0.02 preserves all existing experiments.
Recording, training and evaluation share this setting. See
[BC task-difficulty calibration](bc_task_difficulty.md) before using broader
resets with expert data; old narrow-reset demos are not matched-distribution data.

| Component | Standard Lift | Easy variant |
| --- | --- | --- |
| Cube XY spawn (m) | x in [0.4, 0.6], y in [-0.25, 0.25] | x in [0.48, 0.52], y in [-0.02, 0.02] |
| Cube initial center height (m) | 0.055, then settles | 0.021, the measured resting height |
| Goal in robot frame (m) | x in [0.4, 0.6], y in [-0.25, 0.25], z in [0.25, 0.5] | fixed (0.5, 0, 0.2) |
| Robot reset | Native pose, open fingers | Downward open pre-grasp; fingertip-center frame near (0.499, 0, 0.092) |

The cube stays on the table, ungrasped and outside the success region. The easy
reset removes native startup lift credit: the height-based reward threshold is
0.04 m, below the native airborne spawn but above the resting cube. Reward
definitions and weights themselves are unchanged.

The seven reset joint angles were calibrated against the installed Isaac asset.
They are written by [a reset event](../src/adapters/lift_reset.py), after native
reset events, without changing `default_joint_pos`. Thus the action mapping
`q_target = q_default + 0.5 * action`, policy action bounds, relative-joint
observations, actuator gains, gravity, and dynamics remain unchanged. The
gripper opens; velocities are zeroed. No hidden simulation steps, IK controller,
demonstrations, or compensating controller are added to training.

Native PD control has gravity sag: commanding the reset joint angles as actuator
targets does not hold the same physical pose. Geometry validation uses a holding
action with target offset `gravity_torque / stiffness`, through the existing
position-action interface. This diagnostic is not used by the learners. Also,
zero arm action still commands the native default pose, not the new reset pose.
The variant changes the initial state, not the initial policy's action prior.

## Run either algorithm on the same variant

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/pmd_easy.yaml

python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/ppo_easy.yaml
```

PMD retains UTD 2, 32 complete episodes and 32 interleaved improvements per
batch, the same successor critic, and quantile 0.25. PPO retains its native
4,096 environments, 24-step rollouts, architecture and update rule. Both use
fixed environment rewards without curriculum. Inline validation is disabled;
use separate-process checkpoint evaluation to keep collection uninterrupted.

### Tuned PPO and maximum-entropy successor-PMD sweep

Both `ppo_easy.yaml` and `pmd_easy.yaml` use discount `0.98` and entropy rate
`alpha = 0.006`. The common objective is the entropy-regularized MDP
`sum_t gamma^t * dt * (reward_rate_t + alpha * entropy_t)`. The environment's
`step_dt` is applied inside both implementations; at `dt = 0.02`, the
coefficient on one step's `-log pi` is `0.00012`.

Successor features remain reward- and policy-entropy-independent. The separate
scalar `Q_H` critic has its own online network, Polyak target, Adam optimizer,
and replay draw. Its target is
`gamma * mask * (-dt * log pi(a'|s') + Q_H_target(s',a'))`; control uses
`Q_reward + alpha * Q_H`. PMD policy improvement separately includes the
current action's `-dt * alpha * log pi` term because `Q_H(s,a)` starts at the
next policy action. PPO adds the sampled term to rollout rewards before GAE and
value targets, with the backend's optimizer-only entropy loss disabled to avoid
double counting. This is a fixed entropy rate, not automatic temperature tuning.

PPO's clipping, architecture, parallelism and rollout length remain unchanged.
Earlier entropy-free comparison artifacts keep their frozen configs
and must not be relabeled as maximum-entropy runs.

Run the successor-PMD variants with this override (replace `0.25` with `0.5`
or `0.75`):

```bash
python scripts/train.py --task Isaac-Lift-Cube-Franka-v0 --headless \
  --config configs/lift_cube_franka/pmd_easy.yaml \
  --set algo.pmd.value.successor.control_quantile=0.25
```

This is the percentile across **scalarized successor-Q heads**, not a change
to successor TD targets or a separate optimistic exploration actor. With the
current three heads, sorted as `q1 <= q2 <= q3`, linear interpolation gives:

| Quantile | Control score | Interpretation |
| --- | --- | --- |
| 0.25 | `(q1 + q2) / 2` | Lower quartile; pessimistic |
| 0.5 | `q2` | Median, not the ensemble mean |
| 0.75 | `(q2 + q3) / 2` | Upper quartile; optimistic |
| 0.9 | `0.2 q2 + 0.8 q3` | Near-maximum; stronger optimism |
| 1.0 | `q3` | Maximum head; most optimistic |

Keep seed, reward, reset distribution, networks, Adam settings, replay and
update schedule fixed across those three arms. Evaluate immutable checkpoints
in separate simulator processes on the same held-out seeds. Plot success
against training interactions, and also total interactions including evaluation;
report completed training and evaluation trajectories separately. For PMD,
one batch is 32 complete trajectories and at most 8,000 retained transitions.
Physical interactions can exceed retained transitions after early terminations.
PPO's trajectory count records observed episode endings; initial staggered
episodes may be partial. Equal wall time is not an equal sample budget,
especially when the high-UTD arms share a GPU. A single training seed is a
screening result, not a replicated claim of superiority.

Report easy-task performance separately from transfer to the original task:
use `--set env.variant=standard` for the latter. Reward weights, 250-step
horizon, termination rules, and 8-cm success tolerance remain unchanged. The
comparison's sustained-goal metric still requires success throughout the last
25 steps. Do not compare easy-trained PMD to old standard-task PPO as though
their training tasks matched.

For IRL, demonstrations must match the initial-state and goal distribution
(or the occupancy comparison needs explicit distribution correction).

## Verification and artifacts

Unit tests check preset isolation, unchanged control/rewards, selected-lane
resets, preserved defaults, no RNG consumption, strict task selection, and
both experiment configs. Real-Isaac reset checks use 32 environments and seeds
1001/1002: verify cube bounds, command position, no free lift/success, finite
physics, and actuator offsets. Calibration and diagnostics are recorded under
`logs/benchmarks/lift_easy_20260919_dDwm8C/`; their interactions are diagnostic
work, not training samples.
