"""Video-recording helpers for the train_irl script.

Wraps a gym ``RecordVideo`` around the training env per the requested mode
(interval-by-step, interval-by-iteration, or single final clip), and handles
the post-training cleanup (renaming step-indexed clips to iteration indices,
copying the final clip into ``<log_dir>/videos/``).

Public surface (used by ``train_irl.py``):
    VideoSetup, setup_video_recording, copy_final_videos, rename_iteration_files
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Callable


__all__ = [
    "VideoSetup",
    "setup_video_recording",
    "copy_final_videos",
    "rename_iteration_files",
]


@dataclass(slots=True)
class VideoSetup:
    """Video-recording state needed for post-training cleanup."""

    record_video_wrapper: Any = None
    train_video_dir: str | None = None
    video_dir_for_iteration_names: str | None = None
    video_steps_per_iteration: int | None = None


def setup_video_recording(
    env: Any,
    *,
    args_cli: argparse.Namespace,
    log_dir: str,
    max_iterations: int,
    steps_per_env_per_cycle: int,
    gym_wrappers: Any,
) -> tuple[Any, VideoSetup]:
    """Wrap ``env`` with gym's ``RecordVideo`` per the requested video mode.

    Returns the wrapped env and a ``VideoSetup`` carrying the state the
    post-training cleanup needs (final-clip wrapper, source dir, optional
    iteration-rename hint).
    """
    video_setup = VideoSetup()
    if not args_cli.video:
        return env, video_setup

    train_video_dir = os.path.join(log_dir, "videos", "train")
    video_setup.train_video_dir = train_video_dir

    if args_cli.video_final_only:
        # Final clip(s) are produced by an explicit post-training rollout
        # (see runner.record_final_rollouts), not by an in-training trigger.
        # Install a never-firing trigger and disable auto-stop (video_length=0)
        # so recording boundaries are fully manual.
        wrapped = gym_wrappers.RecordVideo(
            env,
            video_folder=train_video_dir,
            step_trigger=lambda step_idx: False,
            video_length=0,
            disable_logger=True,
        )
        video_setup.record_video_wrapper = wrapped
        return wrapped, video_setup

    step_trigger, rename_dir, steps_per_iter = _build_video_step_trigger(
        video_final_only=False,
        video_interval_iterations=(
            int(args_cli.video_interval_iterations)
            if args_cli.video_interval_iterations is not None
            else None
        ),
        video_interval=int(args_cli.video_interval),
        max_iterations=int(max_iterations),
        steps_per_iter=int(steps_per_env_per_cycle),
        train_video_dir=train_video_dir,
    )
    wrapped = gym_wrappers.RecordVideo(
        env,
        video_folder=train_video_dir,
        step_trigger=step_trigger,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    video_setup.video_dir_for_iteration_names = rename_dir
    video_setup.video_steps_per_iteration = steps_per_iter
    return wrapped, video_setup


def copy_final_videos(
    args_cli: argparse.Namespace,
    *,
    log_dir: str,
    video: VideoSetup,
) -> None:
    """Copy the final/most-recent training clip(s) into ``<log_dir>/videos/``."""
    if not args_cli.video or video.train_video_dir is None:
        return
    videos_dir = os.path.join(log_dir, "videos")
    if args_cli.video_final_only:
        # Dual-clip mode: deterministic + stochastic, named at the source by
        # runner.record_final_rollouts.
        for mode_name in ("inference", "stochastic"):
            source = os.path.join(video.train_video_dir, f"final_{mode_name}.mp4")
            if not os.path.isfile(source):
                continue
            final_path = os.path.join(videos_dir, f"final_{mode_name}.mp4")
            try:
                shutil.copy2(source, final_path)
                print(f"[INFO] Copied final {mode_name} rollout video to {final_path}")
            except Exception as exc:
                print(f"[WARN] Failed to copy final {mode_name} video locally: {exc}")
        return

    # Interval modes: copy the latest trigger-produced clip.
    try:
        latest_mp4 = _latest_mp4_in(video.train_video_dir)
        if latest_mp4 is not None:
            final_path = os.path.join(videos_dir, "final.mp4")
            shutil.copy2(latest_mp4, final_path)
            print(f"[INFO] Copied final rollout video to {final_path}")
    except Exception as exc:
        print(f"[WARN] Failed to copy final video locally: {exc}")


def rename_iteration_files(*, video_dir: str, steps_per_iteration: int) -> None:
    """Rewrite ``*-step-N.mp4`` filenames in ``video_dir`` to ``*-iter-K.mp4``.

    Public wrapper around the internal implementation so ``train_irl.py``'s
    finally-block doesn't reach into a private name. A no-op when ``video_dir``
    doesn't exist.
    """
    _rename_video_files_to_iteration_index(video_dir, steps_per_iteration)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _latest_mp4_in(video_dir: str) -> str | None:
    """Highest-numbered ``*-iter-K.mp4`` (or ``*-step-N.mp4``) in ``video_dir``."""
    if not os.path.isdir(video_dir):
        return None
    mp4s = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not mp4s:
        return None
    pat = re.compile(r"-(?:iter|step)-(\d+)\.mp4$")
    latest = max(
        mp4s,
        key=lambda name: (
            int(pat.search(name).group(1)) if pat.search(name) else -1,
            name,
        ),
    )
    return os.path.join(video_dir, latest)


def _build_video_step_trigger(
    *,
    video_final_only: bool,
    video_interval_iterations: int | None,
    video_interval: int,
    max_iterations: int,
    steps_per_iter: int,
    train_video_dir: str,
) -> tuple[Callable[[int], bool], str | None, int | None]:
    """Build the gym ``RecordVideo.step_trigger`` for the requested mode.

    Returns ``(step_trigger, rename_dir, steps_per_iter)`` where the last two
    are non-None only for the iteration-aligned modes (final-only and
    interval-by-iterations). For those modes, the caller post-processes
    ``rename_dir`` to rewrite ``*-step-N.mp4`` filenames to ``*-iter-K.mp4``.
    The pure-step interval mode returns ``(trigger, None, None)``.
    """
    if video_final_only:
        # Trigger fires at the start of the last iteration so the clip captures
        # the final rollout. `video_length` may exceed `steps_per_iter`; gym's
        # RecordVideo keeps recording across env.step() calls.
        final_trigger_step = max(0, (max_iterations - 1) * steps_per_iter)
        return (
            (lambda step_idx: step_idx == final_trigger_step),
            train_video_dir,
            steps_per_iter,
        )

    if video_interval_iterations is not None:
        if video_interval_iterations <= 0:
            raise ValueError(
                f"`--video_interval_iterations` must be > 0, got {video_interval_iterations}."
            )
        interval_steps = video_interval_iterations * steps_per_iter
        return (
            (lambda step_idx: step_idx % interval_steps == 0),
            train_video_dir,
            steps_per_iter,
        )

    if video_interval <= 0:
        raise ValueError(f"`--video_interval` must be > 0, got {video_interval}.")
    return (lambda step_idx: step_idx % video_interval == 0), None, None


def _rename_video_files_to_iteration_index(
    video_dir: str, steps_per_iteration: int
) -> None:
    if steps_per_iteration <= 0:
        raise ValueError(
            f"`steps_per_iteration` must be > 0, got {steps_per_iteration}."
        )
    if not os.path.isdir(video_dir):
        return

    step_name_pattern = re.compile(r"^(?P<prefix>.+)-step-(?P<step>\d+)(?P<ext>\.mp4)$")
    for file_name in sorted(os.listdir(video_dir)):
        match = step_name_pattern.match(file_name)
        if match is None:
            continue

        step_idx = int(match.group("step"))
        iter_idx = step_idx // steps_per_iteration
        new_name = f"{match.group('prefix')}-iter-{iter_idx}{match.group('ext')}"

        source_path = os.path.join(video_dir, file_name)
        target_path = os.path.join(video_dir, new_name)
        if source_path == target_path:
            continue
        os.replace(source_path, target_path)
