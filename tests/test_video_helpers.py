from __future__ import annotations

import argparse

from scripts._video_helpers import finalize_videos


def test_finalize_videos_promotes_deterministic_and_stochastic(tmp_path):
    log_dir = tmp_path / "run"
    train_video_dir = log_dir / "videos" / "train"
    train_video_dir.mkdir(parents=True)
    (train_video_dir / "final_inference.mp4").write_bytes(b"inference")
    (train_video_dir / "final_stochastic.mp4").write_bytes(b"stochastic")
    args = argparse.Namespace(video=True, video_final_only=True)

    finalize_videos(
        args,
        log_dir=str(log_dir),
        train_video_dir=str(train_video_dir),
        rename_dir=None,
        rename_steps_per_iter=None,
    )

    assert (log_dir / "videos" / "final_inference.mp4").read_bytes() == b"inference"
    assert (log_dir / "videos" / "final_stochastic.mp4").read_bytes() == b"stochastic"
