"""Construct one training run: environment, reward, algorithm, runner.

Everything Isaac-specific about starting a run lives here. The pipeline calls
these builders in order and owns nothing but the lifecycle.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable

import torch

from adapters import (
    RslRlPpoAdapter,
    RslRlVecEnvAdapter,
    build_off_policy_stack,
    load_actor_only_checkpoint,
    make_actor_critic,
)
from adapters.pmd import PmdAdapter
from algorithms import IrlAlgorithm, build_bc_alg
from algorithms.on_policy.ppo_bc import make_ppo_with_bc_cls
from config.schema import PpoAlgoCfg, TrainCfg
from data import install_expert_features, load_expert_success_rate
from interfaces import FeatureSource, FeatureSpec, RlAlgorithm, RuntimeContext
from reward import DenseFeatureRewardModel, LinearFeatureRewardModel
from reward.features import FeatureColumns, ManagerBasedFeatureSource
from reward.fixed import FixedLinearReward
from reward.lift_success import add_success_bonus_term, build_success_fn
from reward.scorer import (
    AdditiveLinearRewardScorer,
    LearnedLinearRewardScorer,
    LearnedRewardScorer,
    RewardScorer,
)
from reward.terminal_success import TERMINAL_SUCCESS_TERM
from runner import Runner
from training.env_variants import configure_lift_easy
from training.run_output import extract_ground_truth_reward_weights


def drop_reward_curriculum(env_cfg: Any) -> None:
    """Remove curriculum terms that rewrite reward weights mid-run.

    Training measures progress against a fixed reward, and Isaac Lab's
    manipulation tasks do not supply one: Lift multiplies its action_rate and
    joint_vel penalties by 1000 once common_step_counter passes 10,000
    environment steps, validation rollouts included. Off-policy arms reach that
    step count around 140k transitions while PPO at 4096 environments would not
    reach it until 41M, so two runs of the same task would be scored against
    two different reward functions. The pre-reset tap refuses to run against
    weights that change, so this must happen before the environment is built.
    """
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is None:
        return
    for name, term in list(vars(curriculum).items()):
        func = getattr(term, "func", None)
        if getattr(func, "__name__", "") == "modify_reward_weight":
            setattr(curriculum, name, None)


def build_env_cfg(
    cfg: TrainCfg, *, parse_env_cfg: Callable[..., Any], use_fabric: bool
):
    """Parse and adjust the task config before the simulator sees it."""
    env_cfg = parse_env_cfg(
        cfg.env.name,
        device=cfg.env.device,
        num_envs=cfg.env.num_envs,
        use_fabric=use_fabric,
    )
    env_cfg.seed = int(cfg.seed)
    env_cfg.is_finite_horizon = cfg.env.is_finite_horizon
    if cfg.env.variant == "lift_easy":
        configure_lift_easy(env_cfg, cube_xy_half_range=cfg.env.lift_cube_xy_half_range)
    if not cfg.env.reward_curriculum:
        drop_reward_curriculum(env_cfg)
    if cfg.env.add_success_bonus:
        added = add_success_bonus_term(
            env_cfg,
            threshold=float(cfg.env.success_threshold),
            weight=float(cfg.env.success_bonus_weight),
        )
        if not added:
            raise ValueError(
                f"env.add_success_bonus is set but task {cfg.env.name!r} has no "
                "`rewards` section to add the term to."
            )
    if cfg.env.terminal_success_reward:
        add_terminal_success_task(env_cfg, cfg)
    return env_cfg


def add_terminal_success_task(env_cfg: Any, cfg: TrainCfg) -> None:
    """Turn Lift into the sparse finite-horizon task the study is defined on.

    Every dense term keeps its place in the reward manager at weight zero, so
    it is still the feature vector phi the reward learner fits, but the
    environment itself now pays exactly one thing: a unit on the final
    transition of an episode that held the goal. The student is told where in
    the episode it is, because a finite-horizon value function is not
    stationary and no amount of critic capacity recovers a clock it cannot see.
    """
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg

    from adapters.lift_terminal import LiftTerminalSuccess, episode_context

    if cfg.env.name != "Isaac-Lift-Cube-Franka-v0":
        raise ValueError(
            "env.terminal_success_reward currently supports Isaac-Lift-Cube-Franka-v0 "
            f"only, got {cfg.env.name!r}."
        )
    if TERMINAL_SUCCESS_TERM in cfg.feature_map.ignored_reward_terms:
        raise ValueError(
            f"feature_map.ignored_reward_terms drops {TERMINAL_SUCCESS_TERM!r}, but "
            "it is the only term this task pays for, so the reward the policy "
            "maximizes would have no feature column to sit in."
        )
    for term in vars(env_cfg.rewards).values():
        if isinstance(term, RewardTermCfg):
            term.weight = 0.0
    setattr(
        env_cfg.rewards,
        TERMINAL_SUCCESS_TERM,
        RewardTermCfg(
            func=LiftTerminalSuccess,
            weight=1.0,
            params={
                "threshold": float(cfg.env.success_threshold),
                "hold_steps": int(cfg.env.success_hold_steps),
            },
        ),
    )
    # Lift reports one observation group, so the actor and the critic read the
    # same tensor; a task that splits them has to be told about both.
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.episode_context = ObservationTermCfg(func=episode_context)


def forced_feature_terms(cfg: TrainCfg, *, env: Any) -> set[str]:
    """Reward terms that stay features although their weight is zero.

    A zero-weight term is normally not a feature: switching a term off should
    make it disappear, not leave a silent column behind. The terminal-success
    task zeroes every dense term on purpose, and those terms are exactly the
    phi being learned from, so they have to be asked back in by name.
    """
    forced = set(cfg.feature_map.force_include_terms)
    if not cfg.env.terminal_success_reward:
        return forced
    rewards = env.unwrapped.cfg.rewards
    return forced | {
        str(name)
        for name, term in vars(rewards).items()
        if term is not None and float(getattr(term, "weight", 1.0)) == 0.0
    }


def learned_feature_columns(
    cfg: TrainCfg, feature_source: FeatureSource
) -> FeatureColumns | None:
    """Which collected columns the reward learner fits, or None when unnamed.

    Explicit named scales select a basis; otherwise preserve the original
    convention of learning every dense term but not terminal success.
    """
    names = feature_source.spec.names
    if names is None:
        if cfg.reward.learned_feature_scales is not None:
            raise ValueError(
                "Named learned feature scales require named collected features."
            )
        return None
    scales = cfg.reward.learned_feature_scales
    if scales is not None:
        missing = set(scales) - set(names)
        if missing:
            raise ValueError(
                f"Learned reward features {sorted(missing)} are not collected; "
                f"available features are {list(names)}."
            )
        return FeatureColumns.select(names, [name for name in names if name in scales])
    if not cfg.env.terminal_success_reward:
        return FeatureColumns.identity(names)
    return FeatureColumns.select(
        names, [name for name in names if name != TERMINAL_SUCCESS_TERM]
    )


def learned_feature_spec(cfg: TrainCfg, feature_source: FeatureSource) -> FeatureSpec:
    """The feature space the reward model is defined over."""
    columns = learned_feature_columns(cfg, feature_source)
    return feature_source.spec if columns is None else columns.spec()


def build_reward_model(
    cfg: TrainCfg, *, env: Any, feature_source: FeatureSource, device: str
) -> LinearFeatureRewardModel | DenseFeatureRewardModel | None:
    """Build the reward model the run will train, or fix to the env's weights."""
    model_class = (
        LinearFeatureRewardModel
        if cfg.reward.model.is_linear
        else DenseFeatureRewardModel
    )
    if not cfg.reward.learns_a_reward:
        return None
    spec = learned_feature_spec(cfg, feature_source)
    if model_class is LinearFeatureRewardModel:
        scales = cfg.reward.learned_feature_scales
        ordered_scales = (
            None if scales is None else [scales[name] for name in spec.names]
        )
        model = model_class(
            cfg.reward.model, num_features=spec.dim, feature_scales=ordered_scales
        ).to(device)
    else:
        model = model_class(cfg.reward.model, num_features=spec.dim).to(device)
    return model


def build_reward_scorer(
    cfg: TrainCfg,
    *,
    env: Any,
    feature_source: FeatureSource,
    reward_model: LinearFeatureRewardModel | DenseFeatureRewardModel | None,
) -> RewardScorer | None:
    """The scorer the policy maximizes; ``None`` means the env's own reward."""
    if cfg.reward.source == "environment":
        return None
    if reward_model is None:
        raise RuntimeError("A learned reward needs a reward model.")
    if not isinstance(reward_model, LinearFeatureRewardModel):
        if cfg.reward.source == "environment_plus_learned":
            raise ValueError(
                "reward.source=environment_plus_learned requires a linear reward "
                "model: the two rewards are added as coefficients on one basis."
            )
        return LearnedRewardScorer(
            reward_model, feature_dim=int(feature_source.spec.dim)
        )

    names = feature_source.spec.names
    if names is None:
        raise ValueError("A linear learned reward needs named reward features.")
    step_dt = float(getattr(env.unwrapped, "step_dt", 1.0))
    learned = LearnedLinearRewardScorer(
        reward_model,
        feature_names=learned_feature_spec(cfg, feature_source).names,
        step_dt=step_dt,
    )
    if cfg.reward.source == "learned" and learned.feature_names == names:
        return learned
    # The critics score whole collected feature vectors, so a learned reward
    # over a narrower basis has to be completed on every remaining column by a
    # coefficient the reward learner cannot move: the task's own weight when
    # the environment reward is part of the objective, and zero when it is not.
    fixed = FixedLinearReward(
        names,
        _fixed_reward_weights(cfg, env=env, names=names),
        step_dt=step_dt,
    ).to(cfg.device)
    return AdditiveLinearRewardScorer(
        fixed,
        learned,
        fixed_coefficient=cfg.reward.environment_weight,
        learned_coefficient=cfg.reward.learned_weight,
    )


def _fixed_reward_weights(
    cfg: TrainCfg, *, env: Any, names: tuple[str, ...]
) -> torch.Tensor:
    """The immutable part of an additive linear reward, in feature order. [D]."""
    if cfg.reward.source != "environment_plus_learned":
        return torch.zeros(len(names))
    weights = extract_ground_truth_reward_weights(env)
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(
            f"The environment exposes no reward weight for {missing}, so the "
            "fixed half of reward.source=environment_plus_learned is undefined."
        )
    return torch.tensor([float(weights[name]) for name in names])


def build_algorithm(
    cfg: TrainCfg,
    *,
    env: Any,
    observations: Any,
    obs_groups: dict[str, list[str]],
    feature_source: FeatureSource,
    reward: RewardScorer | None,
    bc_alg: Any,
    deps: Any,
) -> RlAlgorithm:
    """Build the configured policy algorithm, wired to its reward."""
    if isinstance(cfg.algorithm, PpoAlgoCfg):
        return _build_ppo(
            cfg,
            env=env,
            observations=observations,
            obs_groups=obs_groups,
            feature_source=feature_source,
            reward=reward,
            bc_alg=bc_alg,
            deps=deps,
        )
    stack = build_off_policy_stack(
        env=env,
        observations=observations,
        algorithm_cfg=cfg.algorithm,
        ignored_reward_terms=cfg.feature_map.ignored_reward_terms,
        force_include_reward_terms=forced_feature_terms(cfg, env=env),
        device=cfg.device,
        seed=cfg.seed,
        reward=reward,
        bc=bc_alg,
        init_policy_checkpoint=cfg.init_policy_checkpoint,
    )
    return stack.algorithm


def _build_ppo(
    cfg: TrainCfg,
    *,
    env: Any,
    observations: Any,
    obs_groups: dict[str, list[str]],
    feature_source: FeatureSource,
    reward: RewardScorer | None,
    bc_alg: Any,
    deps: Any,
) -> RslRlPpoAdapter:
    from dataclasses import asdict

    actor, critic = make_actor_critic(
        obs=observations,
        obs_groups=obs_groups,
        action_dim=int(env.action_dim),
        cfg=cfg.actor_critic,
        device=cfg.device,
        MLPModel=deps.MLPModel,
    )
    if cfg.init_policy_checkpoint is not None:
        load_actor_only_checkpoint(
            actor, os.path.abspath(cfg.init_policy_checkpoint), device=cfg.device
        )
    storage = deps.RolloutStorage(
        "rl",
        int(env.num_envs),
        int(cfg.runner.steps_per_env_per_cycle),
        observations,
        [int(env.action_dim)],
        cfg.device,
    )
    ppo_class = make_ppo_with_bc_cls(deps.PPO) if bc_alg is not None else deps.PPO
    extra = {"bc_alg": bc_alg} if bc_alg is not None else {}
    ppo_kwargs = asdict(cfg.algorithm)
    entropy_rate = float(ppo_kwargs.pop("entropy_rate"))
    # Entropy enters sampled rollout rewards below. Disable the backend's
    # optimizer-side bonus so the same entropy term is not counted twice.
    ppo = ppo_class(
        actor=actor,
        critic=critic,
        storage=storage,
        device=cfg.device,
        entropy_coef=0.0,
        **ppo_kwargs,
        **extra,
    )
    return RslRlPpoAdapter(
        ppo,
        env=env,
        feature_source=feature_source if cfg.reward.learns_a_reward else None,
        reward=reward,
        steps_per_env_per_cycle=int(cfg.runner.steps_per_env_per_cycle),
        entropy_rate=entropy_rate,
        step_dt=float(getattr(env.unwrapped, "step_dt", 1.0)),
        device=cfg.device,
        bc_alg=bc_alg,
    )


def build_reward_learner(
    cfg: TrainCfg,
    *,
    env: Any,
    feature_source: FeatureSource,
    reward_model: Any,
) -> IrlAlgorithm | None:
    """Build the reward learner and fill its expert buffer, when one is wanted."""
    if not cfg.reward.learns_a_reward:
        return None
    if reward_model is None:
        raise RuntimeError("Learning a reward needs a reward model.")
    expert_success_rate = load_expert_success_rate(
        os.path.abspath(cfg.expert_data.path)
    )
    learner = IrlAlgorithm(
        reward=reward_model,
        feature_spec=learned_feature_spec(cfg, feature_source),
        gamma=cfg.gamma,
        step_dt=float(env.unwrapped.step_dt),
        cfg=cfg.irl,
        device=cfg.device,
        imitator_buffer_cfg=cfg.runner.imitator_buffer,
        expert_buffer_cfg=cfg.runner.expert_buffer,
        expert_success_rate=expert_success_rate,
        environment_reward_weights=_environment_reward_weights(
            env, names=learned_feature_spec(cfg, feature_source).names
        ),
    )
    install_expert_features(
        learner,
        cfg.expert_data.path,
        max_num_trajectories=cfg.expert_data.num_trajectories,
        subset_strategy=cfg.expert_data.subset_strategy,
        seed=cfg.seed,
    )
    return learner


def _environment_reward_weights(
    env: Any, *, names: tuple[str, ...] | None
) -> torch.Tensor | None:
    """The task's own reward weights, in ``names`` order, when it exposes them.

    Reported, never trained on: it is the only direct read on whether a learned
    reward is recovering the task's own. On the sparse task every dense weight
    is zero, so the comparison is vacuous and the caller reports nothing.
    """
    if names is None:
        return None
    weights = extract_ground_truth_reward_weights(env)
    if not weights:
        return None
    return torch.tensor(
        [float(weights.get(name, 0.0)) for name in names], dtype=torch.float32
    )


def build_bc(cfg: TrainCfg, *, ctx: RuntimeContext) -> Any:
    """Build the BC regularizer, or nothing at all when its weight is zero."""
    return build_bc_alg(
        cfg=cfg.bc,
        expert_data_path=cfg.expert_data.path,
        ctx=ctx,
        device=cfg.device,
        expert_num_trajectories=cfg.expert_data.num_trajectories,
        expert_subset_strategy=cfg.expert_data.subset_strategy,
        seed=cfg.seed,
    )


def build_runner(
    cfg: TrainCfg,
    *,
    env: Any,
    rl_alg: RlAlgorithm,
    reward_learner: IrlAlgorithm | None,
    feature_source: FeatureSource,
    log_dir: str,
    logger: Any,
) -> Runner:
    return Runner(
        env,
        rl_alg=rl_alg,
        reward_learner=reward_learner,
        feature_source=feature_source,
        cfg=cfg.runner,
        success_fn=build_success_fn(
            task_name=str(cfg.env.name),
            threshold=float(cfg.env.success_threshold),
            probe_env=env,
        ),
        log_dir=log_dir,
        device=cfg.device,
        logger=logger,
        episode_evaluator=rl_alg.evaluate_episodes
        if isinstance(rl_alg, PmdAdapter)
        else None,
        reward_features=learned_feature_columns(cfg, feature_source),
    )


def build_feature_source(cfg: TrainCfg, *, env: Any) -> ManagerBasedFeatureSource:
    """The collected feature vector, which is what replay and the critics carry."""
    return ManagerBasedFeatureSource(
        env,
        cfg=replace(
            cfg.feature_map, force_include_terms=forced_feature_terms(cfg, env=env)
        ),
        device=cfg.device,
    )


def wrap_env(env: Any, *, RslRlVecEnvWrapper: Any, seed: int) -> RslRlVecEnvAdapter:
    """Wrap a gym env for rsl_rl, then for this repository's action_dim contract."""
    wrapped = RslRlVecEnvWrapper(env)
    wrapped.seed(int(seed))
    return RslRlVecEnvAdapter(wrapped)


def fix_reward_to_environment_weights(
    reward_model: LinearFeatureRewardModel, *, env: Any, feature_source: FeatureSource
) -> None:
    """Pin a linear reward model at the environment's own reward weights."""
    if feature_source.spec.names is None:
        raise ValueError("Fixing reward weights needs named reward features.")
    weights = extract_ground_truth_reward_weights(env)
    if not weights:
        raise ValueError(
            "The environment exposes no reward-manager weights to fix the model at."
        )
    reward_model.set_named_weights(feature_source.spec.names, weights)
