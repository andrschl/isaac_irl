"""Programmatic W&B report for the fixed-reward RL benchmark."""

from __future__ import annotations


def create_cartpole_overview_report(
    *,
    entity: str,
    project: str,
    group: str,
    title: str,
) -> str:
    """Create a compact report and return its public W&B URL."""
    import wandb_workspaces.reports.v2 as wr

    runset = wr.Runset(
        entity=entity,
        project=project,
        name=group,
        filters=f"Config('group') = '{group}'",
    )
    variant = wr.Config("variant")
    transitions = "Data/simulator_transitions_total"

    report = wr.Report(
        entity=entity,
        project=project,
        title=title,
        description=(
            "Curated sample-efficiency view. All performance plots use deterministic "
            "evaluation and simulator transitions on the x-axis."
        ),
        width="fluid",
        blocks=[
            wr.H1("Performance"),
            wr.P(
                "Use return per step for horizon-independent comparisons. "
                "Curves are grouped by algorithm with standard-deviation bands."
            ),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    _mean_line(
                        wr,
                        title="Deterministic return per step",
                        x=transitions,
                        y="Overview/return_per_step",
                        groupby=variant,
                        layout=wr.Layout(x=0, y=0, w=12, h=8),
                    ),
                    _mean_line(
                        wr,
                        title="Task success rate",
                        x=transitions,
                        y="Overview/success_rate",
                        groupby=variant,
                        layout=wr.Layout(x=12, y=0, w=12, h=8),
                    ),
                    _mean_line(
                        wr,
                        title="Running return AUC",
                        x=transitions,
                        y="Overview/return_per_step_auc",
                        groupby=variant,
                        layout=wr.Layout(x=0, y=8, w=12, h=8),
                    ),
                    _mean_line(
                        wr,
                        title="Best deterministic return per step",
                        x=transitions,
                        y="Overview/best_return_per_step",
                        groupby=variant,
                        layout=wr.Layout(x=12, y=8, w=12, h=8),
                    ),
                ],
            ),
            wr.H1("Learning health"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    _mean_line(
                        wr,
                        title="Critic loss",
                        x=transitions,
                        y="Overview/critic_loss",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Actor parameter change",
                        x=transitions,
                        y="Overview/actor_parameter_delta",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Policy standard deviation",
                        x=transitions,
                        y="Overview/policy_std",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Policy updates per rollout",
                        x=transitions,
                        y="Overview/policy_updates",
                        groupby=variant,
                    ),
                ],
            ),
            wr.H1("PMD projection"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    _mean_line(
                        wr,
                        title="Realized actor KL",
                        x=transitions,
                        y="Overview/pmd_actor_kl",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Candidate target KL",
                        x=transitions,
                        y="Overview/pmd_target_kl",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Candidate effective sample size",
                        x=transitions,
                        y="PMD/effective_sample_size",
                        groupby=variant,
                    ),
                ],
            ),
            wr.H1("Safety checks"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    _mean_line(
                        wr,
                        title="Action clipping fraction (PPO only; tanh cannot clip)",
                        x=transitions,
                        y="Overview/action_clipped_fraction",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Action boundary saturation",
                        x=transitions,
                        y="Overview/action_saturation_fraction",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Time-limit fraction (bootstrapped, not terminal)",
                        x=transitions,
                        y="Overview/timeout_fraction",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Reward reconstruction error (<1e-5)",
                        x=transitions,
                        y="Overview/reward_reconstruction_error",
                        groupby=variant,
                    ),
                    _mean_line(
                        wr,
                        title="Non-finite count (must be zero)",
                        x=transitions,
                        y="Overview/non_finite_count",
                        groupby=variant,
                    ),
                ],
            ),
        ],
    )
    return str(report.save().url)


def _mean_line(wr, *, title, x, y, groupby, layout=None):
    """Build one unsmoothed mean curve with a seed standard-deviation band."""
    return wr.LinePlot(
        title=title,
        x=x,
        y=[y],
        groupby=groupby,
        groupby_aggfunc="mean",
        groupby_rangefunc="stddev",
        smoothing_type="none",
        aggregate=True,
        layout=layout or wr.Layout(w=8, h=7),
    )
