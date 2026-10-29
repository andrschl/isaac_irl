import torch

from reward.episode_success import episode_success


def test_terminal_success_uses_final_valid_step_and_failure_takes_precedence():
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.bool)
    features = torch.zeros(4, 3, 2)
    features[:, :, 0] = 1.0  # legacy bonus must not override the terminal rule
    features[0, -1, 1] = 50.0
    features[1, 1:, 1] = 50.0  # padding / subsequent auto-reset episode
    features[2, -1, 1] = 50.0  # simultaneous failure and timeout
    features[3, 0, 1] = 50.0  # not the final transition
    terminated = torch.zeros_like(mask)
    terminated[1, 0] = terminated[2, -1] = True
    truncated = torch.zeros_like(mask)
    truncated[:, -1] = True
    success = episode_success(
        features,
        ("success_bonus", "terminal_success"),
        mask,
        terminated=terminated,
        truncated=truncated,
    )
    assert success.tolist() == [True, False, False, False]


def test_continuing_success_bonus_keeps_ever_reached_semantics():
    features = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]])
    mask = torch.tensor([[True, True], [True, False]])
    flags = torch.zeros_like(mask)
    success = episode_success(
        features, ("success_bonus",), mask, terminated=flags, truncated=flags
    )
    assert success.tolist() == [True, False]
    assert (
        episode_success(features, ("other",), mask, terminated=flags, truncated=flags)
        is None
    )
