"""Video-recording helpers shared by training scripts.

Wraps `gym.wrappers.RecordVideo` step-trigger construction, post-run
filename rewriting (``-step-N.mp4`` → ``-iter-K.mp4`` when the trigger is
iteration-aligned), and the post-run copy of the final/latest clip into
``<log_dir>/videos/``. None of this depends on Isaac or rsl_rl — pure
filesystem + gym wrapper logic.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from typing import Callable


def latest_mp4_in(video_dir: str) -> str | None:
    """Highest-numbered ``*-iter-K.mp4`` (or ``*-step-N.mp4``) in ``video_dir``."""
    if not os.path.isdir(video_dir):
        return None
    mp4s = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not mp4s:
        return None
    pat = re.compile(r"-(?:iter|step)-(\d+)\.mp4$")
    latest = max(mp4s, key=lambda name: (int(pat.search(name).group(1)) if pat.search(name) else -1, name))
    return os.path.join(video_dir, latest)


def build_video_step_trigger(
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
        return (lambda step_idx: step_idx == final_trigger_step), train_video_dir, steps_per_iter

    if video_interval_iterations is not None:
        if video_interval_iterations <= 0:
            raise ValueError(
                f"`--video_interval_iterations` must be > 0, got {video_interval_iterations}."
            )
        interval_steps = video_interval_iterations * steps_per_iter
        return (lambda step_idx: step_idx % interval_steps == 0), train_video_dir, steps_per_iter

    if video_interval <= 0:
        raise ValueError(f"`--video_interval` must be > 0, got {video_interval}.")
    return (lambda step_idx: step_idx % video_interval == 0), None, None


def rename_video_files_to_iteration_index(video_dir: str, steps_per_iteration: int) -> None:
    if steps_per_iteration <= 0:
        raise ValueError(f"`steps_per_iteration` must be > 0, got {steps_per_iteration}.")
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


def finalize_videos(
    args_cli: argparse.Namespace,
    *,
    log_dir: str | None,
    train_video_dir: str | None,
    rename_dir: str | None,
    rename_steps_per_iter: int | None,
) -> None:
    """Post-training video bookkeeping: rename step-indexed files into
    iteration-indexed ones (iteration-aligned modes only) and copy the
    final/latest clip into ``<log_dir>/videos/``.

    Safe to call unconditionally — no-ops when the relevant inputs are None.
    Best-effort: never raises; logs warnings on failure (videos must never
    break a training run).
    """
    if rename_dir is not None and rename_steps_per_iter is not None:
        try:
            rename_video_files_to_iteration_index(
                video_dir=rename_dir, steps_per_iteration=rename_steps_per_iter
            )
        except Exception as exc:
            print(f"[WARN] Failed to rename video files with iteration indices: {exc}")

    if not (args_cli.video and log_dir is not None and train_video_dir is not None):
        return

    videos_dir = os.path.join(log_dir, "videos")
    if args_cli.video_final_only:
        # Explicit dual-clip mode: deterministic + stochastic, named at the
        # source by runner.record_final_rollouts.
        for mode_name in ("inference", "stochastic"):
            source = os.path.join(train_video_dir, f"final_{mode_name}.mp4")
            if not os.path.isfile(source):
                continue
            final_path = os.path.join(videos_dir, f"final_{mode_name}.mp4")
            try:
                shutil.copy2(source, final_path)
                print(f"[INFO] Copied final {mode_name} rollout video to {final_path}")
            except Exception as exc:
                print(f"[WARN] Failed to copy final {mode_name} video locally: {exc}")
    else:
        # Interval modes: copy the latest trigger-produced clip.
        try:
            latest_mp4 = latest_mp4_in(train_video_dir)
            if latest_mp4 is not None:
                final_path = os.path.join(videos_dir, "final.mp4")
                shutil.copy2(latest_mp4, final_path)
                print(f"[INFO] Copied final rollout video to {final_path}")
        except Exception as exc:
            print(f"[WARN] Failed to copy final video locally: {exc}")
