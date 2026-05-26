from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from algorithms.bc import BC, BCCfg
from algorithms.ppo_with_bc import make_ppo_with_bc_cls
from storage.obs_action_storage import ObsActionBufCfg, ObsActionBuffer
from utils.runtime_context import RuntimeContext


# ---------------------------------------------------------------------------
# ObsActionBuffer
# ---------------------------------------------------------------------------


def _make_buffer() -> ObsActionBuffer:
    ctx = RuntimeContext(num_envs=1, feature_dim=2, device="cpu")
    return ObsActionBuffer(cfg=ObsActionBufCfg(), ctx=ctx)


def test_obs_action_buffer_load_and_sample_round_trip():
    buf = _make_buffer()
    ep1 = (
        {"policy": {"x": torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)}},
        torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2),
    )
    ep2 = (
        {"policy": {"x": torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 100}},
        torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2) + 100,
    )
    buf.load_episodes([ep1, ep2])
    assert len(buf) == 9
    assert buf.action_dim == 2

    obs_mb, act_mb = buf.sample(batch_size=4, device=torch.device("cpu"))
    assert act_mb.shape == (4, 2)
    assert obs_mb["policy"]["x"].shape == (4, 3)


def test_obs_action_buffer_rejects_mismatched_T():
    buf = _make_buffer()
    bad = (
        {"policy": {"x": torch.zeros(5, 3)}},
        torch.zeros(4, 2),  # mismatch
    )
    with pytest.raises(ValueError, match="first dim must equal"):
        buf.load_episodes([bad])


def test_obs_action_buffer_rejects_leaf_structure_mismatch():
    buf = _make_buffer()
    ep1 = ({"policy": {"x": torch.zeros(3, 2)}}, torch.zeros(3, 2))
    ep2 = ({"policy": {"y": torch.zeros(3, 2)}}, torch.zeros(3, 2))
    with pytest.raises(ValueError, match="leaf structure mismatch"):
        buf.load_episodes([ep1, ep2])


# ---------------------------------------------------------------------------
# BC loss
# ---------------------------------------------------------------------------


class _FakeMLPActor(nn.Module):
    """Stand-in for rsl_rl 5.x MLPModel with a state-independent Gaussian head."""

    def __init__(self, in_dim: int, action_dim: int) -> None:
        super().__init__()
        self.mean_net = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._distribution: torch.distributions.Normal | None = None

    def _features(self, obs):
        return obs["policy"]["x"] if isinstance(obs, dict) else obs

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        del masks, hidden_state
        mean = self.mean_net(self._features(obs))
        if not stochastic_output:
            return mean
        std = self.log_std.exp().expand_as(mean)
        self._distribution = torch.distributions.Normal(mean, std)
        return self._distribution.sample()

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions).sum(dim=-1)


def _make_bc_setup(loss_type: str) -> tuple[BC, _FakeMLPActor]:
    buf = _make_buffer()
    expert_obs = torch.randn(64, 3)
    expert_actions = (expert_obs @ torch.randn(3, 2)) + 0.1
    buf.load_episodes([({"policy": {"x": expert_obs}}, expert_actions)])

    bc = BC(
        cfg=BCCfg(alpha=1.0, loss_type=loss_type, batch_size=32),
        storage=buf,
        device=torch.device("cpu"),
    )
    actor = _FakeMLPActor(in_dim=3, action_dim=2)
    return bc, actor


@pytest.mark.parametrize("loss_type", ["nll", "mse"])
def test_bc_loss_decreases_under_optimization(loss_type: str):
    torch.manual_seed(0)
    bc, actor = _make_bc_setup(loss_type)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)

    initial = bc.compute_loss(actor).item()
    for _ in range(200):
        opt.zero_grad()
        loss = bc.compute_loss(actor)
        loss.backward()
        opt.step()
    final = bc.compute_loss(actor).item()
    assert final < initial - 0.1, f"BC {loss_type} loss did not decrease (init={initial}, final={final})"


def test_bc_validation_loss_none_without_split():
    bc, actor = _make_bc_setup("mse")
    assert bc.val_storage is None
    assert bc.compute_validation_loss(actor) is None


@pytest.mark.parametrize("loss_type", ["nll", "mse"])
def test_bc_validation_loss_is_float_and_gradless(loss_type: str):
    torch.manual_seed(0)
    train_buf = _make_buffer()
    train_buf.load_episodes([({"policy": {"x": torch.randn(32, 3)}}, torch.randn(32, 2))])
    val_buf = _make_buffer()
    val_buf.load_episodes([({"policy": {"x": torch.randn(16, 3)}}, torch.randn(16, 2))])

    bc = BC(
        cfg=BCCfg(alpha=1.0, loss_type=loss_type, batch_size=8),
        storage=train_buf,
        val_storage=val_buf,
    )
    actor = _FakeMLPActor(in_dim=3, action_dim=2)

    val_loss = bc.compute_validation_loss(actor)
    assert isinstance(val_loss, float)
    # No gradients should have been accumulated on the actor.
    assert all(p.grad is None for p in actor.parameters())


def test_bc_loss_rejects_unknown_loss_type():
    buf = _make_buffer()
    buf.load_episodes([({"policy": {"x": torch.zeros(2, 3)}}, torch.zeros(2, 2))])
    with pytest.raises(ValueError, match="loss_type"):
        BC(cfg=BCCfg(loss_type="bogus"), storage=buf)


def test_bc_alpha_validation():
    buf = _make_buffer()
    buf.load_episodes([({"policy": {"x": torch.zeros(2, 3)}}, torch.zeros(2, 2))])
    with pytest.raises(ValueError, match="alpha"):
        BC(cfg=BCCfg(alpha=1.5), storage=buf)


# ---------------------------------------------------------------------------
# PPOWithBC: convex-combined gradient
# ---------------------------------------------------------------------------


class _FakePPO:
    """Tiny PPO stand-in with a single-minibatch update body.

    We use a quadratic loss `0.5 * ||theta - target||^2` whose gradient w.r.t.
    the actor parameters is `(theta - target)`, then call optimizer.step().
    PPOWithBC should intercept this step, scale our grad by (1-alpha) and
    accumulate alpha*grad_bc on top.
    """

    def __init__(self, actor, target, device="cpu", **kwargs):
        del kwargs
        self.actor = actor
        self.target = target
        self.optimizer = torch.optim.SGD(actor.parameters(), lr=1.0)
        self.max_grad_norm = 1e9  # effectively disable clipping for the test

    def update(self):
        self.optimizer.zero_grad()
        loss = 0.5 * sum(
            ((p - t) ** 2).sum() for p, t in zip(self.actor.parameters(), self.target)
        )
        loss.backward()
        self.optimizer.step()
        return {"surrogate": float(loss.detach().item())}


class _TinyActor(nn.Module):
    """Linear-mean actor with the rsl_rl 5.x MLPModel API surface."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3, 2))
        self._distribution: torch.distributions.Normal | None = None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        del masks, hidden_state
        mean = obs["policy"]["x"] @ self.w
        if not stochastic_output:
            return mean
        std = torch.ones_like(mean)
        self._distribution = torch.distributions.Normal(mean, std)
        return self._distribution.sample()

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(actions).sum(dim=-1)


def _make_full_dataset_buffer(expert_x, expert_actions):
    """Build a buffer that samples without replacement so a batch_size=N
    minibatch is exactly the full dataset (in permuted order). Makes the
    convex-combination test analytically exact."""
    ctx = RuntimeContext(num_envs=1, feature_dim=2, device="cpu")
    buf = ObsActionBuffer(
        cfg=ObsActionBufCfg(sample_with_replacement=False),
        ctx=ctx,
    )
    buf.load_episodes([({"policy": {"x": expert_x}}, expert_actions)])
    return buf


def test_ppo_with_bc_gradient_is_convex_combination():
    """With alpha=0.5, the update should reduce w by 0.5*(g_ppo + g_bc)."""
    torch.manual_seed(0)
    # Expert dataset: action = x @ W_true
    W_true = torch.tensor([[1.0, -2.0], [0.5, 0.3], [-0.7, 1.1]])
    expert_x = torch.randn(64, 3)
    expert_actions = expert_x @ W_true
    N = expert_x.shape[0]
    A = expert_actions.shape[1]
    buf = _make_full_dataset_buffer(expert_x, expert_actions)
    # batch_size = N -> sample() returns a full permutation of the dataset.
    bc = BC(cfg=BCCfg(alpha=0.5, loss_type="mse", batch_size=N), storage=buf)

    policy = _TinyActor()
    # PPO "target" pulls w toward W_target.
    W_target = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    wrapped_cls = make_ppo_with_bc_cls(_FakePPO)
    wrapped = wrapped_cls(actor=policy, target=[W_target], bc_alg=bc)

    # Analytical gradients at w=0:
    #   g_ppo = w - W_target = -W_target
    #   L_bc  = mean over (N*A) of (x w - a)^2  ->  g_bc = (2/(N*A)) X^T (Xw - A)
    #         = -(2/(N*A)) X^T X W_true  at w=0
    g_ppo = -W_target
    g_bc = -(2.0 / (N * A)) * (expert_x.T @ expert_x @ W_true)

    expected_grad = 0.5 * g_ppo + 0.5 * g_bc
    expected_w = -expected_grad  # lr=1, w starts at 0

    wrapped.update()
    actual_w = policy.w.detach()

    assert torch.allclose(actual_w, expected_w, atol=1e-5), (
        f"Convex combination mismatch.\n"
        f"  expected: {expected_w}\n"
        f"  actual:   {actual_w}\n"
        f"  diff:     {actual_w - expected_w}"
    )


def test_ppo_with_bc_alpha_zero_passes_through():
    """alpha=0 should behave exactly like the base PPO update."""
    torch.manual_seed(0)
    W_true = torch.eye(3, 2)
    buf = _make_buffer()
    buf.load_episodes([({"policy": {"x": torch.randn(4, 3)}}, torch.randn(4, 2))])
    bc = BC(cfg=BCCfg(alpha=0.0, loss_type="mse", batch_size=4), storage=buf)

    W_target = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    policy_a = _TinyActor()
    base = _FakePPO(actor=policy_a, target=[W_target])
    base_result = base.update()

    policy_b = _TinyActor()
    wrapped_cls = make_ppo_with_bc_cls(_FakePPO)
    wrapped = wrapped_cls(actor=policy_b, target=[W_target], bc_alg=bc)
    wrapped_result = wrapped.update()

    assert torch.allclose(policy_a.w, policy_b.w, atol=1e-7)
    assert base_result["surrogate"] == wrapped_result["surrogate"]


def test_ppo_with_bc_bc_only_update_runs_bc_alone():
    torch.manual_seed(0)
    W_true = torch.tensor([[1.0, -2.0], [0.5, 0.3], [-0.7, 1.1]])
    expert_x = torch.randn(64, 3)
    expert_actions = expert_x @ W_true
    N = expert_x.shape[0]
    A = expert_actions.shape[1]
    buf = _make_full_dataset_buffer(expert_x, expert_actions)
    bc = BC(cfg=BCCfg(alpha=1.0, loss_type="mse", batch_size=N), storage=buf)

    policy = _TinyActor()
    W_target = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    wrapped_cls = make_ppo_with_bc_cls(_FakePPO)
    wrapped = wrapped_cls(actor=policy, target=[W_target], bc_alg=bc)

    result = wrapped.bc_only_update()
    assert "bc" in result
    # g_bc = -(2/(N*A)) X^T X W_true; one step at lr=1 gives w = -g_bc.
    expected_w = (2.0 / (N * A)) * (expert_x.T @ expert_x @ W_true)
    assert torch.allclose(policy.w.detach(), expected_w, atol=1e-5)


# ---------------------------------------------------------------------------
# HDF5 loader for (obs, actions)
# ---------------------------------------------------------------------------


def test_obs_action_hdf5_loader_round_trip(tmp_path: Path):
    h5py = pytest.importorskip("h5py")

    # Import via path to avoid importing the full Isaac Lab dependency tree.
    import importlib.util
    import sys

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "irl" / "train_irl.py"
    spec = importlib.util.spec_from_file_location("train_irl_for_bc_tests", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_irl_for_bc_tests"] = module
    spec.loader.exec_module(module)

    path = tmp_path / "demos.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        obs_group = demo.create_group("obs")
        policy_group = obs_group.create_group("policy")
        policy_group.create_dataset("x", data=torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3).numpy())
        demo.create_dataset("actions", data=torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2).numpy())

    episodes = module.load_obs_action_episodes(str(path))
    assert len(episodes) == 1
    obs, actions = episodes[0]
    assert actions.shape == (5, 2)
    assert obs["policy"]["x"].shape == (5, 3)
    assert torch.allclose(actions, torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2))
