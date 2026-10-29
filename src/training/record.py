"""Record demonstrations by rolling a trained teacher out on the study's task.

Recording uses the *same* episode collector training uses. That is the whole
design: a demonstration is only expert data if it means what a collected
transition means, and the pre-reset successor, the two ending flags and the
complete final transition are exactly the parts that are easy to get subtly
wrong twice in two different ways.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from adapters import IsaacPreResetTap, load_actor_only_checkpoint
from adapters.episode_collector import EpisodeCollector
from adapters.transition_collector import TransitionCollector
from algorithms.off_policy.actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
    policy_observation_tensor,
)
from config import build_record_parser, load_experiment_cfg, task_slug
from config.schema import PmdCfg, TrainCfg
from data import RobomimicDataCollector
from data.feature_semantics import demonstration_metadata
from reward.terminal_success import TERMINAL_SUCCESS_TERM
from storage.episode_batch import EpisodeBatch
from storage.observations import Observation, map_observation
from training.build import build_env_cfg, forced_feature_terms, wrap_env
from training.pipeline import REPO_ROOT

SUCCESS_FEATURES = (TERMINAL_SUCCESS_TERM, "success_bonus")


@dataclass(slots=True)
class RecordingDeps:
    """Isaac objects, imported only once the simulator app exists."""

    AppLauncher: Any
    gym: Any
    parse_env_cfg: Any
    RslRlVecEnvWrapper: Any
    MLPModel: Any
    resolve_obs_groups: Any


@dataclass(frozen=True, slots=True)
class RecordingCost:
    """What the recording spent, reported next to what it produced.

    A demonstration set is not free data. Attempts that failed, and the vector
    lanes stepped alongside them, are simulator interactions a comparison
    against an online arm has to account for somewhere.

    Attempts that were thrown away are counted in two kinds, because they mean
    opposite things. An attempt that *failed* says the teacher cannot do the
    task; an attempt that succeeded past the quota says it can, and was simply
    not needed. Reporting one number for both reads as a failing teacher.
    """

    environment_transitions: int
    """Every simulator step the recording took, discarded lanes included."""
    attempted_episodes: int
    successful_episodes: int
    written_episodes: int
    horizon: int

    @property
    def kept_transitions(self) -> int:
        return self.written_episodes * self.horizon

    @property
    def failed_episodes(self) -> int:
        return self.attempted_episodes - self.successful_episodes

    @property
    def surplus_episodes(self) -> int:
        """Qualifying attempts past the quota; the teacher, not its failures."""
        return self.successful_episodes - self.written_episodes

    @property
    def success_rate(self) -> float:
        """Share of complete attempts that met the task's own success rule."""
        if self.attempted_episodes == 0:
            return 0.0
        return self.successful_episodes / self.attempted_episodes

    @property
    def discarded_transitions(self) -> int:
        return self.environment_transitions - self.kept_transitions

    def report(self) -> str:
        return (
            f"[INFO] wrote {self.written_episodes} demonstrations from "
            f"{self.attempted_episodes} attempts: {self.successful_episodes} "
            f"succeeded ({self.success_rate:.1%}), {self.failed_episodes} failed, "
            f"{self.surplus_episodes} succeeded past the quota. "
            f"{self.environment_transitions} environment transitions, of which "
            f"{self.kept_transitions} were kept."
        )


def success_feature_index(feature_names: tuple[str, ...]) -> int:
    """Column of the feature that says this step reached the goal.

    One rule decides what counts as a demonstration: the success feature must
    be set on the episode's *final* transition. On the sparse task that is
    literally the reward being paid; on the dense task it means the goal was
    still held when the horizon arrived. Either way an attempt that succeeded
    early and then drifted away is not a demonstration of the task.
    """
    for name in SUCCESS_FEATURES:
        if name in feature_names:
            return feature_names.index(name)
    raise ValueError(
        f"None of {list(SUCCESS_FEATURES)} is a reward feature of this task, so "
        f"there is no success to record against; it reports {list(feature_names)}."
    )


def record_demonstrations(
    *,
    episodes: EpisodeCollector,
    writer: RobomimicDataCollector,
    action_fn: Callable[[Observation], torch.Tensor],
    feature_names: tuple[str, ...],
    num_demos: int,
    horizon: int,
    max_attempted_episodes: int | None = None,
) -> RecordingCost:
    """Collect waves of complete episodes until ``num_demos`` have been kept."""
    if max_attempted_episodes is not None and max_attempted_episodes <= 0:
        raise ValueError("max_attempted_episodes must be positive.")
    environment_transitions = 0
    attempted = 0
    successful = 0
    while writer.written_episodes < num_demos:
        if max_attempted_episodes is not None and (
            attempted + episodes.steps.num_envs > max_attempted_episodes
        ):
            message = (
                f"Recording attempt budget {max_attempted_episodes} cannot fit another "
                f"complete wave of {episodes.steps.num_envs}: {attempted} attempts, "
                f"{successful} successes, {writer.written_episodes}/{num_demos} demos "
                f"written, {environment_transitions} physical interactions. "
                "Partial demonstrations remain on disk; qualify the teacher before increasing the budget."
            )
            # Isaac shutdown can terminate before a pending exception prints.
            print(f"[ERROR] {message}", flush=True)
            raise RuntimeError(message)
        batch = episodes.collect(
            policy_version=0,
            episodes=episodes.steps.num_envs,
            action_fn=action_fn,
        )
        environment_transitions += batch.environment_transitions
        attempted += batch.episode_count
        successful += _write_batch(writer, batch, feature_names=feature_names)
        print(
            f"[INFO] {writer.written_episodes}/{num_demos} demonstrations after "
            f"{attempted} attempts, {successful} of which succeeded "
            f"({environment_transitions} transitions).",
            flush=True,
        )
    return RecordingCost(
        environment_transitions=environment_transitions,
        attempted_episodes=attempted,
        successful_episodes=successful,
        written_episodes=writer.written_episodes,
        horizon=horizon,
    )


def _write_batch(
    writer: RobomimicDataCollector,
    batch: EpisodeBatch,
    *,
    feature_names: tuple[str, ...],
) -> int:
    """Buffer one wave, keep the attempts that succeeded, and count them."""
    transitions = batch.transitions
    success = transitions.features[:, :, success_feature_index(feature_names)] > 0.0
    accepted = batch.mask.all(dim=1) & success[:, -1]
    for step in range(int(batch.mask.shape[1])):
        writer.add("obs", _step_slice(transitions.policy_observations, step))
        writer.add("next_obs", _step_slice(transitions.next_policy_observations, step))
        writer.add("actions", transitions.actions[:, step])
        writer.add("rewards", transitions.environment_rewards[:, step])
        writer.add("terminated", transitions.terminated[:, step])
        writer.add("truncated", transitions.truncated[:, step])
        writer.add(
            "features", _named_features(transitions.features[:, step], feature_names)
        )
        writer.add("success", success[:, step])
    writer.flush(accepted.nonzero(as_tuple=False).reshape(-1))
    # Whatever is still buffered was not written - a failed attempt, or one
    # more success than was asked for. Either way it is not a demonstration.
    writer.discard(torch.arange(int(batch.mask.shape[0])))
    return int(accepted.sum())


def _named_features(
    features: torch.Tensor, feature_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """One dataset per reward term, so a file names the space it was recorded in."""
    return {name: features[:, index] for index, name in enumerate(feature_names)}


def _step_slice(observations: Observation, step: int) -> Observation:
    return map_observation(observations, lambda leaf: leaf[:, step])


def teacher_observation_widths(env: Any) -> dict[str, int]:
    """Width of each observation group as the teacher was trained to see it.

    This repository appends an episode clock to the student's observation,
    because a finite-horizon value function is not stationary. A teacher
    trained before that has no weights for the extra columns, so they are cut -
    by reading the per-term widths back off the observation manager rather than
    by assuming where they ended up.
    """
    manager = env.unwrapped.observation_manager
    widths: dict[str, int] = {}
    for group, names in manager.active_terms.items():
        dims = manager.group_obs_term_dim[group]
        widths[str(group)] = sum(
            int(math.prod(dim))
            for name, dim in zip(names, dims, strict=True)
            if name != "episode_context"
        )
    return widths


def _teacher_view(observations: Observation, widths: dict[str, int]) -> Observation:
    """Cut every observation group back to the teacher's own width."""
    if isinstance(observations, torch.Tensor):
        return observations
    return {
        key: value[..., : widths[key]] if key in widths else value
        for key, value in observations.items()
    }


def main(argv: list[str] | None = None, deps: RecordingDeps | None = None) -> None:
    """Build the environment, load the teacher, and write one demo file."""
    app_launcher_cls = deps.AppLauncher if deps is not None else _app_launcher_cls()
    args = build_record_parser(app_launcher_cls).parse_args(argv)
    if args.max_attempted_episodes is not None and args.max_attempted_episodes <= 0:
        raise ValueError("--max_attempted_episodes must be positive.")
    if args.video:
        args.enable_cameras = True

    simulation_app = app_launcher_cls(args).app
    if deps is None:
        deps = _load_deps(app_launcher_cls)
    try:
        _record(args, deps)
    finally:
        simulation_app.close()


def _record(args: argparse.Namespace, deps: RecordingDeps) -> None:
    output_dir = args.output_dir or os.path.join("logs", "demos", task_slug(args.task))
    output_path = os.path.join(os.path.abspath(output_dir), "demos.hdf5")
    cfg = _recording_cfg(args, output_path=output_path)
    env_cfg = build_env_cfg(
        cfg, parse_env_cfg=deps.parse_env_cfg, use_fabric=not args.disable_fabric
    )

    native_env = deps.gym.make(
        args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None
    )
    if args.video:
        native_env = deps.gym.wrappers.RecordVideo(
            native_env,
            video_folder=os.path.join(output_dir, "videos"),
            step_trigger=lambda step: step == 0,
            video_length=int(args.video_length),
            name_prefix="collection",
            disable_logger=True,
        )
    env = wrap_env(
        native_env, RslRlVecEnvWrapper=deps.RslRlVecEnvWrapper, seed=cfg.seed
    )
    horizon = int(env.max_episode_length)
    if args.demo_length is not None and int(args.demo_length) != horizon:
        raise ValueError(
            f"--demo_length {args.demo_length} is not this task's episode length "
            f"{horizon}. Demonstrations are whole episodes; splitting a rollout "
            "into fixed-length chunks would store successors that cross a reset."
        )

    tap = IsaacPreResetTap(
        env,
        ignored_terms=cfg.feature_map.ignored_reward_terms,
        force_include_terms=forced_feature_terms(cfg, env=env),
        device=cfg.device,
        capture_terminal_observations=True,
    )
    feature_names = tap.spec.names or ()
    success_feature_index(feature_names)  # fail before the first simulator step
    writer = RobomimicDataCollector(
        task_name=args.task,
        output_dir=output_dir,
        filename="demos.hdf5",
        num_demos=int(args.num_demos),
        metadata={
            **demonstration_metadata(
                horizon=horizon,
                step_dt=tap.step_dt,
                is_finite_horizon=bool(env_cfg.is_finite_horizon),
                action_bound=_action_bound(cfg),
                feature_names=feature_names,
            ),
            "environment_config": json.dumps(asdict(cfg.env), sort_keys=True),
            "recording_seed": cfg.seed,
            "teacher_checkpoint": os.path.abspath(args.checkpoint),
        },
        overwrite=bool(args.overwrite),
    )
    writer.reset()
    try:
        with tap.installed():
            student = _student_policy(cfg, env=env, device=cfg.device)
            action_fn = _teacher_action_fn(
                args, deps, env=env, cfg=cfg, student=student
            )
            episodes = EpisodeCollector(
                TransitionCollector(
                    env,
                    student,
                    tap,
                    rollout_steps=1,
                    device=cfg.device,
                    seed=cfg.seed,
                    is_finite_horizon=bool(env_cfg.is_finite_horizon),
                ),
                episodes_per_batch=int(env.num_envs),
            )
            cost = record_demonstrations(
                episodes=episodes,
                writer=writer,
                action_fn=action_fn,
                feature_names=feature_names,
                num_demos=int(args.num_demos),
                horizon=horizon,
                max_attempted_episodes=args.max_attempted_episodes,
            )
        print(cost.report(), flush=True)
        Path(output_dir, "recording.json").write_text(
            json.dumps(
                {
                    "environment_config": asdict(cfg.env),
                    "seed": cfg.seed,
                    "teacher_checkpoint": os.path.abspath(args.checkpoint),
                    "teacher_config": args.teacher_config or args.config,
                    "cost": asdict(cost),
                    "success_rate": cost.success_rate,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        writer.close()
        env.close()


def _recording_cfg(args: argparse.Namespace, *, output_path: str) -> TrainCfg:
    """Load the experiment these demonstrations are being recorded for."""
    cfg, _ = load_experiment_cfg(
        args=argparse.Namespace(
            task=args.task,
            config=args.config,
            set=list(args.set),
            num_envs=args.num_envs,
            device=getattr(args, "device", None),
            # The arm's config names the file it will read; recording writes it.
            expert_data_path=output_path,
        ),
        config_root=REPO_ROOT / "configs",
    )
    return cfg


def _action_bound(cfg: TrainCfg) -> float | None:
    return (
        float(cfg.algorithm.policy.action_bound)
        if isinstance(cfg.algorithm, PmdCfg)
        else None
    )


def _student_policy(
    cfg: TrainCfg, *, env: Any, device: str
) -> StateDependentTanhPolicy:
    """The untrained student actor, built only to carry its action bound.

    The collector checks every action against the bound of the policy that
    produced it. Recording actions the student could never output would make
    behavioral cloning fit something it cannot represent, so the teacher is
    held to the student's bound here rather than to its own.
    """
    if not isinstance(cfg.algorithm, PmdCfg):
        raise ValueError(
            "Recording needs the off-policy policy configuration to know the "
            "action bound demonstrations must respect; set algo.type=pmd."
        )
    policy_cfg = cfg.algorithm.policy
    observations = env.get_observations()
    model = StateDependentTanhGaussianActor(
        int(policy_observation_tensor(observations).shape[-1]),
        int(env.action_dim),
        hidden_dims=policy_cfg.hidden_dims,
        action_bound=policy_cfg.action_bound,
        initial_std=policy_cfg.initial_std,
        minimum_log_std=math.log(policy_cfg.minimum_std),
        maximum_log_std=math.log(policy_cfg.maximum_std),
    ).to(device)
    return StateDependentTanhPolicy(model)


def _teacher_action_fn(
    args: argparse.Namespace,
    deps: RecordingDeps,
    *,
    env: Any,
    cfg: TrainCfg,
    student: StateDependentTanhPolicy,
) -> Callable[[Observation], torch.Tensor]:
    """Load the teacher and expose it as the bounded action function to record."""
    teacher_cfg, _ = load_experiment_cfg(
        args=argparse.Namespace(
            task=args.task,
            config=args.teacher_config or args.config,
            set=[],
            num_envs=args.num_envs,
            device=cfg.device,
            expert_data_path=cfg.expert_data.path,
        ),
        config_root=REPO_ROOT / "configs",
    )
    widths = teacher_observation_widths(env)
    observations = env.get_observations()
    teacher = _load_teacher(
        args,
        deps,
        env=env,
        cfg=teacher_cfg,
        observations=_teacher_view(observations, widths),
        device=cfg.device,
    )
    bound = float(student.model.action_bound)

    def act(step_observations: Observation) -> torch.Tensor:
        with torch.no_grad():
            action = teacher(_teacher_view(step_observations, widths))
        # The recorded action is the one the simulator executed, bounded the
        # way the student's own actions are.
        return action.clamp(-bound, bound)

    return act


def _load_teacher(
    args: argparse.Namespace,
    deps: RecordingDeps,
    *,
    env: Any,
    cfg: TrainCfg,
    observations: Observation,
    device: str,
) -> Callable[[Observation], torch.Tensor]:
    """Rebuild the teacher's architecture and load its parameters."""
    from adapters.actor_critic import make_actor

    checkpoint_path = os.path.abspath(args.checkpoint)
    if isinstance(cfg.algorithm, PmdCfg):
        policy_cfg = cfg.algorithm.policy
        model = StateDependentTanhGaussianActor(
            int(policy_observation_tensor(observations).shape[-1]),
            int(env.action_dim),
            hidden_dims=policy_cfg.hidden_dims,
            action_bound=policy_cfg.action_bound,
            initial_std=policy_cfg.initial_std,
            minimum_log_std=math.log(policy_cfg.minimum_std),
            maximum_log_std=math.log(policy_cfg.maximum_std),
        ).to(device)
        load_actor_only_checkpoint(model, checkpoint_path, device=device)
        model.eval()
        return StateDependentTanhPolicy(model).act_inference

    obs_groups = deps.resolve_obs_groups(
        observations, {}, default_sets=["actor", "critic"]
    )
    actor = make_actor(
        obs=observations,
        obs_groups=obs_groups,
        action_dim=int(env.action_dim),
        cfg=cfg.actor_critic,
        device=device,
        MLPModel=deps.MLPModel,
    )
    load_actor_only_checkpoint(actor, checkpoint_path, device=device)
    actor.eval()
    return actor


def _app_launcher_cls() -> Any:
    from isaaclab.app import AppLauncher

    return AppLauncher


def _load_deps(app_launcher_cls: Any) -> RecordingDeps:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.models import MLPModel
    from rsl_rl.utils import resolve_obs_groups

    return RecordingDeps(
        AppLauncher=app_launcher_cls,
        gym=gym,
        parse_env_cfg=parse_env_cfg,
        RslRlVecEnvWrapper=RslRlVecEnvWrapper,
        MLPModel=MLPModel,
        resolve_obs_groups=resolve_obs_groups,
    )
