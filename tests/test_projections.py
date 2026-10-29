# ruff: noqa: E402

"""Projecting a linear reward back onto its norm ball."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reward.projections import project_onto_l2_ball


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
