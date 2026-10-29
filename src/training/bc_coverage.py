"""Repeated-trajectory coverage of BC member sampling versus one BC policy."""

from __future__ import annotations

import torch

from reward.episode_success import episode_success
from storage.observations import actor_observation, map_observation


def coverage_curve(
    successes: torch.Tensor, *, additional_prefixes: tuple[int, ...] = ()
) -> list[dict[str, float | int]]:
    """For success [T,B], report trial success and any-success by initial state.

    T independent complete trajectories are attempted from each of B matched
    initial states. The prefixes 1,2,4,...,T use the *actual* sampled trials;
    no reward-based member selection or estimated independence is involved.
    """
    if successes.ndim != 2 or successes.dtype != torch.bool:
        raise ValueError(
            f"Expected boolean trial success [T,B], got {tuple(successes.shape)} "
            f"and {successes.dtype}."
        )
    trial_count, initial_states = successes.shape
    if trial_count < 1 or initial_states < 1:
        raise ValueError("Coverage requires at least one trial and initial state.")
    prefixes = []
    n = 1
    while n < trial_count:
        prefixes.append(n)
        n *= 2
    prefixes.append(trial_count)
    if any(prefix < 1 or prefix > trial_count for prefix in additional_prefixes):
        raise ValueError("Coverage prefixes must lie within the trial count.")
    prefixes = sorted(set((*prefixes, *additional_prefixes)))
    result = []
    for n in prefixes:
        trials = successes[:n]
        successful_trials = int(trials.sum())
        covered_states = int(trials.any(dim=0).sum())
        result.append(
            {
                "n": n,
                "successful_trials": successful_trials,
                "trial_success_fraction": successful_trials / (n * initial_states),
                "covered_initial_states": covered_states,
                "at_least_one_success_fraction": covered_states / initial_states,
            }
        )
    return result


def aggregate_coverage(
    rows: list[dict], *, additional_prefixes: tuple[int, ...] = ()
) -> dict:
    """Pool matched initial states across reset seeds, preserving trial index."""
    if not rows:
        raise ValueError("Cannot aggregate BC coverage without reset-seed rows.")
    pooled = {}
    for condition in ("baseline", "ensemble"):
        per_seed = [
            torch.tensor(
                [trial["episode_successes"] for trial in row[condition]],
                dtype=torch.bool,
            )
            for row in rows
        ]
        if len({trial_success.shape[0] for trial_success in per_seed}) != 1:
            raise ValueError("Reset seeds must have the same number of BC trials.")
        pooled[condition] = coverage_curve(
            torch.cat(per_seed, dim=1), additional_prefixes=additional_prefixes
        )
    return {
        "initial_states": sum(row["initial_states"] for row in rows),
        **pooled,
    }


def sampled_member_actions(
    observations,
    *,
    members: list,
    member_by_env: torch.Tensor,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Sample bounded actions [N,A] from one fixed BC member per environment.

    ``member_by_env`` [N] is chosen once before a complete episode wave and
    stays fixed at every physical step. It is never selected using reward.
    """
    actor_obs = actor_observation(observations)
    if actor_obs.ndim != 2 or member_by_env.shape != (actor_obs.shape[0],):
        raise ValueError(
            "Expected actor observations [N,O] and member assignments [N], got "
            f"{tuple(actor_obs.shape)} and {tuple(member_by_env.shape)}."
        )
    if len(generators) != len(members):
        raise ValueError("Need one action generator per BC member.")
    if bool(((member_by_env < 0) | (member_by_env >= len(members))).any()):
        raise ValueError("BC member assignment is outside the member index range.")
    actions = actor_obs.new_empty((actor_obs.shape[0], members[0].model.action_dim))
    for member_index, member in enumerate(members):
        env_mask = member_by_env == member_index
        if not bool(env_mask.any()):
            continue
        member_obs = map_observation(observations, lambda leaf: leaf[env_mask])
        actions[env_mask] = member.sample_with_log_probability(
            member_obs, generator=generators[member_index]
        )[0]
    return actions


def sample_member_schedule(
    *, member_count: int, num_envs: int, trials: int, seed: int, mode: str
) -> torch.Tensor:
    """Return member assignments [T,N], fixed per episode and reproducible.

    In without-replacement mode, every environment gets a shuffled cycle of
    all M members before another cycle begins. When T > M, a fresh cycle is
    required; no policy is repeated *within* a cycle.
    """
    if member_count < 2 or num_envs < 1 or trials < 1:
        raise ValueError("Need multiple members, positive environments and trials.")
    if mode == "with_replacement":
        return torch.stack(
            [
                torch.randint(
                    member_count,
                    (num_envs,),
                    generator=torch.Generator().manual_seed(
                        seed + 300_007 + 10_003 * trial_index
                    ),
                )
                for trial_index in range(trials)
            ]
        )
    if mode != "without_replacement":
        raise ValueError(f"Unknown BC member sampling mode: {mode}.")
    generator = torch.Generator().manual_seed(seed + 300_007)
    assignments = torch.empty((trials, num_envs), dtype=torch.long)
    for env_index in range(num_envs):
        for cycle_start in range(0, trials, member_count):
            cycle_length = min(member_count, trials - cycle_start)
            assignments[cycle_start : cycle_start + cycle_length, env_index] = (
                torch.randperm(member_count, generator=generator)[:cycle_length]
            )
    return assignments


@torch.no_grad()
def evaluate_bc_coverage(
    algorithm,
    *,
    env,
    baseline,
    members: list,
    episodes_per_seed: int,
    seeds: list[int],
    trials: int,
    member_sampling: str = "with_replacement",
) -> dict:
    """Run N stochastic episodes per matched start for both BC conditions.

    The baseline draws N trajectories from the same policy. The ensemble draws
    one member per trajectory and fixes that member for the episode. Member
    assignment can use independent draws or shuffled without-replacement
    cycles. Both conditions use the same reset states; action noise is
    independently seeded per trial. No transition enters replay.
    """
    num_envs = algorithm.collector.steps.num_envs
    if len(members) < 2 or trials <= 0 or not seeds:
        raise ValueError("Require multiple BC members, positive trials, and seeds.")
    if episodes_per_seed != num_envs:
        raise ValueError(
            "Paired BC coverage requires one complete episode wave: "
            f"episodes_per_seed={episodes_per_seed}, num_envs={num_envs}."
        )
    if algorithm.replay.transition_count or algorithm.replay.demonstration_count:
        raise ValueError("BC coverage evaluation requires empty replay.")
    names = algorithm.collector.steps.feature_tap.spec.names
    device = algorithm.collector.steps.device
    cycle_prefix = (
        (len(members),)
        if member_sampling == "without_replacement" and trials >= len(members)
        else ()
    )
    rows = []
    interaction_count = 0
    for seed in seeds:
        reference_start = None
        successes = {"baseline": [], "ensemble": []}
        condition_rows = {"baseline": [], "ensemble": []}
        member_schedule = sample_member_schedule(
            member_count=len(members),
            num_envs=num_envs,
            trials=trials,
            seed=seed,
            mode=member_sampling,
        )
        for trial_index in range(trials):
            assigned_members = member_schedule[trial_index].to(device)
            member_generators = [
                torch.Generator(device=device).manual_seed(
                    seed + 500_009 + 10_003 * trial_index + member_index
                )
                for member_index in range(len(members))
            ]
            baseline_generator = torch.Generator(device=device).manual_seed(
                seed + 700_001 + 10_003 * trial_index
            )

            def baseline_actions(observations):
                return baseline.sample_with_log_probability(
                    observations, generator=baseline_generator
                )[0]

            def ensemble_actions(observations):
                return sampled_member_actions(
                    observations,
                    members=members,
                    member_by_env=assigned_members,
                    generators=member_generators,
                )

            for condition, action_fn in (
                ("baseline", baseline_actions),
                ("ensemble", ensemble_actions),
            ):
                env.seed(seed)
                batch = algorithm.collector.collect(
                    policy_version=0,
                    episodes=episodes_per_seed,
                    action_fn=action_fn,
                )
                starts = actor_observation(batch.transitions.policy_observations)[
                    :, 0
                ].cpu()
                if reference_start is None:
                    reference_start = starts
                    max_start_difference = 0.0
                else:
                    if starts.shape != reference_start.shape:
                        raise ValueError(
                            "BC coverage did not receive matching initial states."
                        )
                    max_start_difference = float((starts - reference_start).abs().max())
                    if max_start_difference > 1e-5:
                        raise ValueError(
                            "BC coverage requires paired initial observations; "
                            f"seed={seed}, trial={trial_index}, condition={condition} "
                            f"differed by up to {max_start_difference:.6g}."
                        )
                transitions = batch.transitions
                success = episode_success(
                    transitions.features,
                    names,
                    batch.mask,
                    terminated=transitions.terminated,
                    truncated=transitions.truncated,
                )
                if success is None or success.shape != (episodes_per_seed,):
                    raise ValueError("BC coverage requires per-episode task success.")
                successes[condition].append(success.cpu())
                interaction_count += batch.environment_transitions
                trial_row = {
                    "trial": trial_index,
                    "successes": int(success.sum()),
                    "episode_successes": success.cpu().tolist(),
                    "episode_returns": batch.returns.cpu().tolist(),
                    "environment_interactions": batch.environment_transitions,
                    "retained_transitions": batch.transition_count,
                    "max_initial_observation_difference": max_start_difference,
                }
                if condition == "ensemble":
                    trial_row["member_by_initial_state"] = (
                        assigned_members.cpu().tolist()
                    )
                condition_rows[condition].append(trial_row)
                print(
                    f"[INFO] BC coverage seed={seed} trial={trial_index + 1}/{trials} "
                    f"{condition}: {int(success.sum())}/{episodes_per_seed} successes, "
                    f"start_diff={max_start_difference:.3g}",
                    flush=True,
                )
        rows.append(
            {
                "seed": seed,
                "initial_states": episodes_per_seed,
                "baseline": condition_rows["baseline"],
                "ensemble": condition_rows["ensemble"],
                "baseline_coverage": coverage_curve(
                    torch.stack(successes["baseline"]),
                    additional_prefixes=cycle_prefix,
                ),
                "ensemble_coverage": coverage_curve(
                    torch.stack(successes["ensemble"]),
                    additional_prefixes=cycle_prefix,
                ),
            }
        )
    if algorithm.replay.transition_count or algorithm.replay.demonstration_count:
        raise RuntimeError("BC coverage evaluation inserted data into replay.")
    return {"rows": rows, "evaluation_interactions": interaction_count}


def publish_bc_coverage(result: dict, *, cfg, directory) -> str:
    """Log trial-success and any-success curves against number of attempts."""
    import wandb

    with wandb.init(
        project=cfg.wandb_project,
        name=directory.name,
        group=cfg.experiment_name + "_bc_coverage",
        job_type="bc-coverage-evaluation",
        dir=str(directory),
        tags=["evaluation", "bc-coverage", "paired-resets"],
        config={
            "member_checkpoints": result["member_checkpoints"],
            "baseline_checkpoint": result["baseline_checkpoint"],
            "protocol": result["protocol"],
            "training_interactions": 0,
        },
    ) as run:
        run.define_metric("Coverage/trials", hidden=True)
        run.define_metric("Coverage/*", step_metric="Coverage/trials", step_sync=False)
        for curve_index, baseline_point in enumerate(result["summary"]["baseline"]):
            ensemble_point = result["summary"]["ensemble"][curve_index]
            n = baseline_point["n"]
            if ensemble_point["n"] != n:
                raise ValueError("Baseline and ensemble trial prefixes must match.")
            run.log(
                {
                    "Coverage/trials": n,
                    "Coverage/initial_states": result["summary"]["initial_states"],
                    "Coverage/baseline/trial_success_fraction": baseline_point[
                        "trial_success_fraction"
                    ],
                    "Coverage/baseline/at_least_one_success_fraction": baseline_point[
                        "at_least_one_success_fraction"
                    ],
                    "Coverage/ensemble/trial_success_fraction": ensemble_point[
                        "trial_success_fraction"
                    ],
                    "Coverage/ensemble/at_least_one_success_fraction": ensemble_point[
                        "at_least_one_success_fraction"
                    ],
                    "Samples/training_environment_interactions": 0,
                    "Samples/evaluation_interactions": result[
                        "evaluation_interactions"
                    ],
                }
            )
        print(f"[INFO] BC coverage W&B: {run.url}", flush=True)
        return run.url
