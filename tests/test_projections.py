# ruff: noqa: E402

"""Tests for reward_model.projections."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reward_model.projections import (
    elastic_net_proximal,
    project_onto_l1_ball,
    project_onto_l2_ball,
    soft_threshold,
)


def test_project_onto_l2_ball_inside():
    x = torch.tensor([[1.0, 0.0, 0.0]])
    out = project_onto_l2_ball(x, radius=2.0)
    torch.testing.assert_close(out, x)
    assert out.norm(p=2, dim=-1).item() <= 2.0 + 1e-6


def test_project_onto_l2_ball_outside():
    x = torch.tensor([[4.0, 0.0, 0.0]])
    out = project_onto_l2_ball(x, radius=1.0)
    torch.testing.assert_close(out.norm(p=2, dim=-1), torch.tensor([1.0]))
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0, 0.0]]))


def test_project_onto_l1_ball_inside():
    x = torch.tensor([[0.5, 0.3, 0.2]])
    out = project_onto_l1_ball(x, radius=1.0)
    torch.testing.assert_close(out, x)
    assert out.abs().sum().item() <= 1.0 + 1e-6


def test_project_onto_l1_ball_outside():
    x = torch.tensor([[3.0, 0.0, 0.0]])
    out = project_onto_l1_ball(x, radius=1.0)
    assert out.abs().sum().item() <= 1.0 + 1e-6
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0, 0.0]]))


def test_soft_threshold_zero_lam():
    x = torch.tensor([1.0, -2.0, 0.5])
    out = soft_threshold(x, lam=0.0)
    torch.testing.assert_close(out, x)


def test_soft_threshold_positive_lam():
    x = torch.tensor([2.0, -1.0, 0.3])
    out = soft_threshold(x, lam=0.5)
    torch.testing.assert_close(out[0], torch.tensor(1.5))
    torch.testing.assert_close(out[1], torch.tensor(-0.5))
    torch.testing.assert_close(out[2], torch.tensor(0.0))


def test_elastic_net_proximal():
    x = torch.tensor([2.0, -1.0])
    out = elastic_net_proximal(x, lam1=0.5, lam2=0.5)
    scale = 1.0 / (1.0 + 2.0 * 0.5)
    expected = scale * soft_threshold(x, 0.5)
    torch.testing.assert_close(out, expected)


def test_project_onto_l1_ball_batched():
    x = torch.tensor([[2.0, 0.0], [0.5, 0.5]])
    out = project_onto_l1_ball(x, radius=1.0)
    assert out.shape == x.shape
    assert (out.abs().sum(dim=1) <= 1.0 + 1e-6).all()


def test_project_onto_l2_ball_invalid_radius():
    with pytest.raises(ValueError, match="radius"):
        project_onto_l2_ball(torch.tensor([1.0]), radius=0)
    with pytest.raises(ValueError, match="radius"):
        project_onto_l1_ball(torch.tensor([1.0]), radius=-1)
