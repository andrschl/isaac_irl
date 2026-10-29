# PMD policy and value updates

This is the equation reference for `algo.type: pmd`, with either value backend.
Policy mirror descent (PMD) alternates approximate evaluation of a frozen policy
with a bounded policy projection. It learns action values, directly or through successor features;
**there is no separately learned state-value network in PMD**.
For IRL, BC, and the PPO baseline, see the [algorithm overview](algorithms.md).

## 1. Scope and notation

- $s$: state observations; $a$: action; $s'$: successor observations after
  taking that action. Actor and critic observations can differ; replay stores
  their respective inputs separately.
- $\pi(a\mid s)$: a policy's action probability density. $\pi_\theta$ is the
  trainable actor with parameters $\theta$; $\pi_k$ is `previous_policy`, a
  frozen snapshot for evaluation/projection cycle $k$.
- $Q^\pi(s,a)$: expected discounted reward return after taking $a$ at $s$ and
  following $\pi$ thereafter. $V^\pi(s)$ averages this over the policy's first
  action as well; PMD does not learn a separate network for it.
- $(s_i,a_i,\phi_i,s'_i,m_i)$: replay row $i$. The raw feature vector
  $\phi_i\in\mathbb R^D$ has $D$ reward features, captured after the physical
  step and before auto-reset; $m_i\in\{0,1\}$ says whether to bootstrap.
- $M$: ensemble heads, indexed by $j$ or $l$; $B_c$: critic batch size **per
  head**; $B_\pi$: actor batch size; $K$: candidate actions per actor state.
- $\gamma$: discount on future returns; $\tau$: Polyak interpolation
  coefficient; $\Delta t$: environment step duration; $w\in\mathbb R^D$:
  current linear reward weights.

An **online critic** is the network trained by gradient descent. Its **target
critic** is a separate copy used to compute temporal-difference (TD) training
labels. Superscript $-$ means this target copy, **not negation**:

| Online quantity | Target counterpart | Meaning |
| --- | --- | --- |
| $Q_j(s,a)$ | $Q_j^-(s,a)$ | Scalar action-value prediction from head $j$ |
| $\Psi_j(s,a)$ | $\Psi_j^-(s,a)$ | Vector of discounted feature-return predictions from head $j$ |
| $\vartheta$ | $\vartheta^-$ | Parameters of the online critic and its target copy |

Target parameters receive no gradients. They are nevertheless updated after
each critic step by Polyak averaging, as specified in section 4. Thus a
"frozen target" is not fixed for the entire run. A TD label, written $y$ below,
is a computed training target, not another network.

For a linear reward, the per-transition reward is
$r_w(\phi)=\Delta t\,w^\top\phi$, where $w^\top\phi$ is the weighted feature
sum. Successor predictions are raw discounted feature sums: neither $\Delta t$
nor $(1-\gamma)$ is included in their TD targets.

Let `terminated` indicate a genuine terminal state, and `done` indicate either
a termination or time-limit truncation. The bootstrap mask is
$m_i=1-\mathrm{terminated}_i$, **not** $1-\mathrm{done}_i$.
Time-limit truncations bootstrap from the true pre-reset successor observation;
genuine terminations do not. Early terminations have not been disabled, and a
time limit does not turn these targets into fixed-horizon returns.

Implementation: [TransitionCollector.stack_steps / _successor_observations](../src/adapters/transition_collector.py),
[LearnedLinearRewardScorer.score](../src/reward/scorer.py).

## 2. Critic update

Draw $MB_c$ replay transitions and give each head a disjoint slice $\mathcal B_j$
of size $B_c$. Sampling is uniform with replacement, so source transitions can
repeat across slices. For every sampled row, draw **one next action**
$a'_i\sim\pi_k(\cdot\mid s'_i)$; there is no maximization over next actions or
behavior-policy importance ratio. Replay determines the regression distribution,
which need not match the occupancy of $\pi_k$. Optional demonstration seeding
changes the sampling mixture; it is off in the checked-in experiments.

All TD labels are detached: no gradient flows through their computation.
$\ell$ below is Huber loss with threshold 1, quadratic near zero residual and
linear outside that range. Each critic update takes one Adam step on the
averaged loss, with gradient-norm clipping. The default entropy rate is zero.

### Successor backend: vector TD, scalar control

The desired successor value $\Psi^{\pi_k}$ is the expected discounted feature
sum below. Here $t$ counts steps from the sampled state-action pair, and
$\mathbb E_{\pi_k}$ averages over subsequent actions from $\pi_k$ and environment
transitions. Head $j$ approximates
$\Psi^{\pi_k}(s,a)=\mathbb E_{\pi_k}[\sum_{t\ge0}\gamma^t\phi_t\mid s_0=s,a_0=a]$.

Let $y^\Psi_{j,i}$ be head $j$'s vector TD label for replay row $i$, and $L_\Psi$
the ensemble loss. The index $d$ selects one of the $D$ features; $h_d>0$ is
its loss-normalization scale. With floor $h_{\min}=$ `return_scale_min`, fit
$h_d=\max(h_{\min},\sqrt{\widehat{\mathbb E}[\phi_d^2]}/(1-\gamma))$, where
$\widehat{\mathbb E}$ is the empirical average over all collected features so far.

Each head bootstraps from **its own target head**, without ensemble reduction:

$$
\begin{aligned}
y^\Psi_{j,i}&=\phi_i+\gamma m_i\Psi^-_j(s'_i,a'_i),\\
L_\Psi&=\frac{1}{MB_cD}\sum_{j=1}^M\sum_{i\in\mathcal B_j}\sum_{d=1}^D
\ell\!\left(\frac{\Psi_{j,d}(s_i,a_i)-y^\Psi_{j,i,d}}{h_d}\right).
\end{aligned}
$$

After the initial fit, the scale is refitted when any observed/fitted ratio
exceeds `scale_refit_ratio`. Online and target output layers are rescaled
together to preserve their predictions. The scale conditions the loss, not
the reward.

For control (scoring candidate actions), let $q_j$ be the scalarized value of
online head $j$, and $Q_{\mathrm{control}}$ the aggregate score used by the
actor. The quantile fraction $\rho=$ `control_quantile` lies in $[0,1]$.
Scalarize **each online head first**, then reduce:

$$
q_j(s,a)=\Delta t\,w^\top\Psi_j(s,a),\qquad
Q_{\mathrm{control}}(s,a)=\operatorname{quantile}_{\rho}
\{q_1(s,a),\ldots,q_M(s,a)\}.
$$

`control_quantile` is a fraction: $\rho=0.25$ means the 25th percentile, exactly
the second-smallest value for five heads. The implementation is
`torch.quantile(..., dim=0, interpolation="linear")`; $0$ is the minimum,
$0.5$ the median, and $1$ the maximum. Below $0.5$ the score is pessimistic;
above it the actor is drawn toward actions the ensemble disagrees about.
Note that an offset shared by every candidate at a state cancels in the
softmax below, and a uniform rescaling is absorbed by the bisection on $c$, so
only the part of the shift that varies *across candidates* changes the target. This is not a componentwise feature quantile or a quantile
regression loss. Signed reward weights are applied before sorting. Changing
$w$ changes control scores immediately, but never successor TD targets.

Successor heads share a residual trunk and have separate vector-output heads.
Implementation: [SuccessorActionValue._per_head_losses / mean / fit_feature_scale](../src/algorithms/off_policy/values/backends.py),
[SuccessorFeatureEnsemble / successor_ensemble_td_target](../src/algorithms/off_policy/values/networks.py).

### Direct backend: scalar TD with clipped targets

Let $y^Q_{j,i}$ be the scalar TD label for head $j$ and replay row $i$, and $L_Q$
the ensemble loss. For each training head $j$, draw a fresh subset $S_j$ of
target-head indices without replacement, shared across that head's minibatch.
The default subset size is two:

$$
\begin{aligned}
y^Q_{j,i}&=r_w(\phi_i)+\gamma m_i\min_{l\in S_j}Q^-_l(s'_i,a'_i),\\
L_Q&=\frac{1}{MB_c}\sum_{j=1}^M\sum_{i\in\mathcal B_j}
\ell\!\left(Q_j(s_i,a_i)-y^Q_{j,i}\right).
\end{aligned}
$$

Control uses $Q_{\mathrm{control}}(s,a)=\min_j Q_j(s,a)$ over **all online heads**.
The heads are independent scalar networks. Replay features are rescored using
the current reward on every critic update; this backend also supports nonlinear
rewards. Optional `target_reduction: independent` uses the paired target head
instead, and `control_reduction: mean` changes control only.

Implementation: [DirectActionValue._bootstrap_value / _per_head_losses / mean](../src/algorithms/off_policy/values/backends.py).

### Optional entropy: off by default

With fixed `entropy_rate` $\alpha>0$, the direct target becomes
$y^Q_{j,i}=r_w(\phi_i)+\gamma m_i[\min_{l\in S_j}Q^-_l(s'_i,a'_i)-\Delta t\,\alpha\log\pi_k(a'_i\mid s'_i)]$.
Its Q already contains the entropy return.

Successor TD stays unchanged. A separate scalar **action-value** critic $H(s,a)$
estimates the discounted entropy return starting at the next action. Its target
copy is $H^-(s,a)$, and its TD label $y^H_i$ is
$y^H_i=\gamma m_i[-\Delta t\log\pi_k(a'_i\mid s'_i)+H^-(s'_i,a'_i)]$.
It uses its own replay draw, Huber loss, Adam step, and target network; control
adds $\alpha H(s,a)$ to the scalarized successor quantile. Thus task reward and
entropy are both integrated over physical time. There is no learned entropy rate.

Implementation: [soft_task_q_td_target / entropy_return_td_target](../src/algorithms/off_policy/values/targets.py),
[EntropyReturnCritic.update](../src/algorithms/off_policy/values/backends.py).

## 3. Policy update

The actor outputs a mean vector $\mu_\theta(s)$ and standard-deviation vector
$\sigma_\theta(s)$, each with $A$ action coordinates. Draw a pre-tanh action
$z\in\mathbb R^A$ from their diagonal Gaussian, then squash it using the action
bound $a_{\max}=$ `policy.action_bound`:
$z\sim\mathcal N(\mu_\theta(s),\operatorname{diag}\sigma_\theta^2(s))$,
$a=a_{\max}\tanh z$. Log probabilities include the scaled-tanh Jacobian, with
`1e-6` inside its logarithm for numerical stability. Log standard deviations
are clamped to the configured bounds.

For Franka Lift, $a_{\max}=4$ in native task action units. The arm commands
$q_{\mathrm{target}}=q_{\mathrm{default}}+0.5a_{\mathrm{arm}}$;
these are absolute joint-position targets, not increments from the current
joint positions. The gripper uses the sign of its action. See the
[action-bound diagnostic](lift_action_bounds.md) for the evidence and limits
of this task-specific choice. Cartpole retains $a_{\max}=1$.

### Frozen candidate target

Draw $B_\pi$ replay states and $K$ actions $a_{ib}\sim\pi_k(\cdot\mid s_i)$
per state, with candidate index $b=1,\ldots,K$. Evaluate them with the **online**
$Q_{\mathrm{control}}$ from section 2. Because this action is fixed by the
candidate, $H(s,a)$ starts at the next policy action; include the current
entropy contribution explicitly:

$$
\widetilde Q(s,a)=Q_{\mathrm{control}}(s,a)-\Delta t\,\alpha\log\pi_k(a\mid s).
$$

Let $\pi_*$ be the intended improved
policy density, and $p_{ib}$ its normalized probability weight on candidate $b$
at state $i$. The old-policy exponent $\beta$ is 1 by default; 0 gives a
Boltzmann target. The nonnegative coefficient $c$ controls how strongly Q
tilts the target toward higher-value candidates:

$$
\begin{aligned}
\pi_*(a\mid s)&\propto\pi_k(a\mid s)^\beta e^{c\widetilde Q(s,a)},\\
p_{ib}&=\operatorname{softmax}_{b}\!\left[
(\beta-1)\log\pi_k(a_{ib}\mid s_i)+c\widetilde Q(s_i,a_{ib})\right].
\end{aligned}
$$

Softmax normalizes the candidate weights so that $\sum_b p_{ib}=1$ for each
state. One shared $c$ is chosen for the whole batch by bounded bisection.
Let $D_{\mathrm{candidate}}$ be the Kullback-Leibler (KL) divergence of these
weights from uniform weights, averaged over states and measured in nats. Its
budget is $\epsilon_{\mathrm{candidate}}=$ `candidate_target_kl`:

$$
D_{\mathrm{candidate}}=\frac{1}{B_\pi}\sum_i\sum_{b=1}^K
p_{ib}\log(Kp_{ib})\le\epsilon_{\mathrm{candidate}}.
$$

This is categorical KL to uniform candidate indices, not an exact continuous
policy KL. The search returns `q_coefficient_min` if its KL already meets or
exceeds the budget, so that endpoint can exceed it; if the maximum is feasible,
it returns the maximum. Otherwise it bisects and retains a feasible coefficient.
Q is not standardized. Additive statewise shifts cancel; at $\beta=1$, positive
rescaling can be compensated by $c$ when the search bounds permit it.

### Weighted projection and trust region

Keep candidate actions, their pre-tanh latents, and $p_{ib}$ fixed throughout
the projection. The actor loss $L_{\mathrm{PMD}}$ is weighted negative log
likelihood, minimized with respect to actor parameters $\theta$:

$$
L_{\mathrm{PMD}}(\theta)=-\frac{1}{B_\pi}\sum_i\sum_{b=1}^K
p_{ib}\log\pi_\theta(a_{ib}\mid s_i).
$$

The code stores weights $Kp_{ib}$ and averages over $B_\pi K$ entries, which is
the same loss. Gradients flow only through the fitted actor's log probability,
not through Q, candidate actions, or weights. This is weighted maximum
likelihood, not a gradient through the critic with respect to actor actions.

Take up to `projection_steps` Adam steps with gradient-norm clipping. After
each step, measure the latent-Gaussian KL **from reference to fitted actor**
on the same replay states. Write $\mu_k,\sigma_k$ for the reference policy's
Gaussian parameters, and $\mu_\theta,\sigma_\theta$ for the fitted actor's;
their state argument is omitted below. Let $D_\mu$ denote the mean-movement
part and $D_\sigma$ the spread-change part of this KL. With $\mathbb E_i$ the
average over sampled states and $r=1,\ldots,A$ the action coordinate:

$$
\begin{aligned}
D_\mu&=\mathbb E_i\sum_{r=1}^A\frac{(\mu_{k,r}-\mu_{\theta,r})^2}{2\sigma_{\theta,r}^2},\\
D_\sigma&=\mathbb E_i\sum_{r=1}^A\left[
\log\frac{\sigma_{\theta,r}}{\sigma_{k,r}}
+\frac{\sigma_{k,r}^2}{2\sigma_{\theta,r}^2}-\frac12\right].
\end{aligned}
$$

The full-KL budget is $\epsilon_{\mathrm{actor}}=$ `actor_projection_kl`; the
spread-only budget is $\epsilon_{\mathrm{spread}}=$ `actor_spread_projection_kl`.
Require $D_\mu+D_\sigma\le\epsilon_{\mathrm{actor}}$ and
$D_\sigma\le\epsilon_{\mathrm{spread}}$. Under the common invertible tanh
transform, the Gaussian KL equals the bounded-policy KL mathematically;
these are batch-average constraints, not guarantees over all states.

If either bound is exceeded, halve the **accumulated displacement from the
projection's starting parameters**, up to `projection_backtrack_steps` times,
then end the projection. If none is feasible, restore the starting parameters.
Adam moments are retained even when parameters are backtracked.

Optional BC mixes in the [expert loss](algorithms.md#behavioral-cloning); its
default weight is zero. The same trust-region checks apply to a mixed update.

Implementation: [Pmd._candidate_batch / _projection_loss / _project / _backtrack_into_trust_region](../src/algorithms/off_policy/pmd.py),
[adaptive_exponential_target](../src/algorithms/off_policy/exponential_target.py),
[decomposed_policy_kl / mix_actor_loss](../src/algorithms/off_policy/common.py),
[StateDependentTanhGaussianActor / StateDependentTanhPolicy](../src/algorithms/off_policy/actor.py).

## 4. Update order and target synchronization

1. Collect `train.episodes_per_batch` complete episodes with a frozen actor.
   Put valid transitions in replay and update feature-scale statistics. If IRL
   is enabled, reuse that same episode batch and update the reward **before**
   the RL updates. Reward weights stay fixed during those RL updates.
2. Wait for `replay.learning_starts`, then run `initial_critic_updates` with the
   actor frozen. Then run the regular schedule, in that same eligible batch.
3. Resolve the critic budget: either `train.critic_updates_per_batch`, or
   `train.critic_updates_per_transition` times newly retained transitions
   (carrying fractional update credit between batches). Divide these steps across
   `policy_improvements_per_batch` evaluate-then-project groups (default one).
   Each group evaluates its frozen $\pi_k$.
   After **each** critic optimizer step, update its target parameters:
   $\vartheta^-\leftarrow(1-\tau)\vartheta^-+\tau\vartheta$.
4. Project the actor, then replace `previous_policy` with a frozen **hard copy**
   of the accepted actor: $\pi_{k+1}\leftarrow\pi_\theta$. This is not Polyak
   averaging. Repeat the group with this new reference policy.

Targets start as exact frozen copies of the online critics. For a configured
budget of $C$ critic steps and $P$ projections, group $g\in\{0,\ldots,P-1\}$
takes $\lfloor C/P\rfloor + \mathbf{1}[g < C\bmod P]$ critic steps.
Require $1\le P\le C$. Exactly $C$ regular critic updates and $P$ projections
execute. With a fixed budget, $C$ is independent of episode length; with UTD,
$C$ scales with retained data. The one-time critic warm-up is additional.
See [episode training](episode_training.md)
for masks, validation boundaries, sample accounting, and the PPO benchmark.

Implementation: [Runner._run_cycle](../src/runner/runner.py),
[PmdAdapter._warm_up_if_needed / update](../src/adapters/pmd.py),
[polyak_update](../src/algorithms/off_policy/values/parameter_ops.py),
[Pmd.critic_update / actor_update](../src/algorithms/off_policy/pmd.py).

## 5. Checked-in experiment settings

These combine the [Franka Lift YAML](../configs/lift_cube_franka/experiment.yaml)
and [Cartpole YAML](../configs/cartpole/experiment.yaml) with
[schema defaults](../src/config/schema.py). CLI overrides can change them.
Keys below are relative to the selected `algo.pmd` block unless explicitly
qualified. These settings were checked on 2026-09-19.

| Setting | Franka Lift | Cartpole |
| --- | --- | --- |
| `value.type` | `successor` | `successor` |
| `value.num_heads` | 3 | 5 |
| Control reduction | `control_quantile: 0.25` | `control_quantile: 0.25` |
| TD head selection | Paired target head | Paired target head |
| `train.gamma` / `train.target_polyak` | 0.99 / 0.005 | 0.99 / 0.005 |
| `train.critic_batch_size` / `actor_batch_size` | 256 / 512 | 256 / 256 |
| `policy_improvements_per_batch` | 1 | 1 |
| `train.episodes_per_batch` / `train.critic_updates_per_batch` | 32 / 1000 | 32 / 200 |
| `train.replay.learning_starts` / `train.initial_critic_updates` | 4096 / 1000 | 1024 / 200 |
| Candidate / actor / spread KL budgets | 0.15 / 0.05 / 0.002 | 0.05 / 0.02 / 0.002 |
| `policy.action_bound` | 4.0 | 1.0 |
| `reward.source` (root key) | Learned linear | Environment |

Both use $\beta=1$, $\alpha=0$, $K=32$; up to 8 projection steps and 8 backtracking halvings;
actor and critic learning rates `3e-4`; maximum gradient norm `1.0`. The
coefficient search uses 32 bisections between `1e-6` and `1e3`. Latent standard
deviations start at `1.0` and are bounded to `[0.05, 2.0]`.
Successor scale floor/refit ratio are `1.0 / 2.0`. BC and periodic resets are off.

The opt-in `configs/lift_cube_franka/pmd_high_reuse.yaml` experiment overrides
the Lift schedule with UTD 2 and 32 interleaved improvements per batch, using
fixed environment reward. It leaves the critic architecture and quantile alone;
the default configs above remain available as conservative baselines.
