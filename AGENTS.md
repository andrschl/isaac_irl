# AGENTS.md

## Purpose
This repository values **readability, explicit tensor semantics, and clean modular design** over clever shortcuts.

Write code that is easy to review, debug, and extend in PyTorch/RL/IRL pipelines.

---

## Core Principles
- Prefer **clear data contracts** (shape, dtype, device) over implicit behavior.
- Prefer **small focused classes/functions** over multi-mode objects with many flags.
- Prefer **explicit validation** (`ValueError`) over silent assumptions or runtime `assert`.
- Keep **feature extraction**, **reward modeling**, **storage**, and **training logic** separate.
- Don't try to get backwards compatibility with old code if it would make new code less clear. Refactor old code when needed.

---

## Tensor Style (High Priority)

### Shape annotations
Document tensor shapes in public APIs and nontrivial helpers.

Examples:
- `z: [N, D]`
- `feats: [B, T, D]`
- `mask: [B, T]`
- `actions: [N, A]`

Use consistent symbols:
- `N` = number of envs
- `B` = batch size
- `T` = trajectory length
- `D` = feature dimension
- `A` = action dimension
- `O` = observation dimension

### Prefer `einops` when it improves readability
Use `einops.rearrange`, `einops.repeat`, and `einops.reduce` for nontrivial reshaping/reordering.

Prefer:
- `rearrange(x, "b t d -> (b t) d")`
when it makes tensor intent clearer than a chain of `reshape/permute/unsqueeze`.

### Avoid ambiguous shape mutation
- Avoid bare `.squeeze()` (it can remove unintended dimensions).
- Prefer `.squeeze(-1)` or `einops.rearrange(...)`.
- If using `view/reshape/permute`, keep it shape-safe and readable; add a short shape comment if nontrivial.

### Masks and fancy indexing are good
Use masks, fancy indexing, and list comprehensions **when they improve clarity**.

- Boolean masks for filtering are encouraged.
- Advanced indexing is fine and often elegant.
- For **nontrivial broadcasted/mixed advanced indexing**, add a short comment with:
  - intended selection semantics (paired vs Cartesian)
  - resulting shape

Example comment style:
- `# select Cartesian product of batch_idx [I] and time_idx [J] -> [I, J]`

### Neat indexing (important)
- Favor indexing that aligns visually with documented tensor shape order.
- Use named intermediate index variables (`batch_idx`, `time_idx`, `env_idx`) instead of overloaded `i/j/k` in larger scopes.
- Prefer explicit slicing (`x[:, :t]`) when it is simpler than advanced indexing.

---

## Naming Conventions
- Use descriptive names in public code: `feature_dim`, `episode_lengths`, `discounted_returns`.
- Short names (`n`, `d`, `t`) are fine only in small local blocks with obvious context.
- Name methods by data contract and intent:
  - `sample_episodes`
  - `sample_discounted_feature_returns`
  - `get_reward_from_features`
- Avoid vague names like `process`, `handle`, `compute_data`.

---

## Config, Context, and State
Keep these distinct:
- **Config**: experiment choices (e.g., `capacity_steps`, `store_dtype`)
- **Runtime context**: derived runtime values (e.g., `num_envs`, `feature_dim`, `device`)
- **State**: mutable internals (buffers, counters, caches)

Shared hyperparameters (e.g. `gamma`) should come from a central config and be passed explicitly to components that use them.

Do not duplicate runtime values across unrelated configs unless necessary.

---

## Class and Module Design
- One class should have **one storage/model semantics**.
- Avoid mode flags that change the meaning/type of internal fields.
- If two modes store fundamentally different things, prefer **two classes**.
- Constructors (`__init__`) should initialize dependencies/state, not perform hidden I/O.

Prefer composition over inheritance in training code:
- `Runner` owns `rl_alg`, `irl_alg`, `feature_map`, buffers, logger
instead of deep inheritance trees.

---

## Validation and Error Messages
- Validate shapes/devices/dtypes at API boundaries.
- Raise actionable errors that include:
  - expected shape/value
  - actual shape/value
  - short fix hint when useful

Example:
- `Expected z shape [num_envs, feature_dim], got ...`

---

## PyTorch Device / Dtype Rules
- Be explicit about storage vs compute device.
- Move tensors at clear boundaries (e.g., buffer insert / sample).
- Avoid hidden device transfers deep inside helpers.
- If using reduced precision for storage (e.g., `float16`), document where conversion occurs.

---

## RL / IRL Project Conventions
- Feature maps are functions/classes of the **env** (`feature_map(env)`), not `(obs, actions)`.
- Reward models consume **features directly** (`[N, D]` or `[B, T, D]`).
- Reward models should not own feature extraction logic.
- Buffers should store **feature trajectories** in explicit episode form unless there is a strong reason otherwise.
- Prefer trajectory batches as `feats [B, T, D]` + `mask [B, T]` for reward learning.

---

## Training Loop Readability
Structure loops into clear blocks:
1. rollout collection
2. algorithm update
3. reward update (if applicable)
4. logging / checkpointing

Keep logging/formatting code separate from core learning logic.

---

## Preferred Refactoring Behavior
- Preserve behavior unless explicitly asked to change it.
- If refactoring, improve readability first, then reduce duplication.
- Do not introduce clever one-liners that obscure tensor semantics.
- Extract small helpers when shape logic repeats.

---

## Preferred Output Style for New Code
- Short docstrings with shape contracts
- Clear variable names
- Small helpers for validation / reshaping
- `einops` for nontrivial tensor transforms
- Explicit indexing and shape-safe operations