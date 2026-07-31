# Phase 8 — GPTQ Error Compensation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPTQ-style off-diagonal error compensation to the shared-codebook product quantizer and measure whether it beats uniform PQ at matched footprint.

**Architecture:** Quantize groups of `sub_dim` input channels in order; after each group, push its reconstruction error onto the not-yet-quantized columns, weighted by the Cholesky factor of the damped inverse input covariance. Compensation changes only *which codes get chosen*, never what is stored, so footprint is identical to the uniform encode by construction. Three rounds alternate codebook refitting with a compensation pass replayed from the original weights.

**Tech Stack:** Python 3.13, PyTorch (CPU for unit tests, CUDA on the box), typer CLI, pytest.

**Spec:** [`docs/specs/2026-07-31-gptq-compensation-phase8-design.md`](../specs/2026-07-31-gptq-compensation-phase8-design.md)
**Prior:** [`docs/weighting-diagnosis.md`](../weighting-diagnosis.md) — why the reweighting family is closed

## Global Constraints

- Branch `feat/gptq-compensation-phase8`, off `main`. Never commit to `main`; the phase lands as one PR.
- Conventional commits (`feat:`, `test:`, `docs:`). No `Co-Authored-By` trailers.
- Type hints on every new signature, params **and** return, with explicit generics: `dict[int, torch.Tensor]`, never bare `dict`.
- No leading underscores on module-level functions, classes, or constants.
- The CI surface is torch-only (`pyproject.toml` `[dependency-groups] test`). No test may import `transformers` or `datasets`.
- Absent a Hessian, every existing code path must stay **byte-identical** — Phases 5/6/7 encodes must not move.
- Test fixtures use **heterogeneous columns and explicitly correlated inputs**. An i.i.d. fixture tests the one regime where these methods are inert (the Phase-7 lesson).
- Run `PYTHONPATH=src .venv/bin/python -m pytest -q` before each commit. Baseline is **57 passed**.

### Existing interfaces this plan builds on

```python
# src/smart_quant/codebook.py
assign(x, centroids, dim_weight=None, weighted_x=None) -> Tensor        # (n,) long
lloyd_kmeans(x, k, iters=10, dim_weight=None) -> tuple[Tensor, Tensor]  # (centroids, idx)
pq_quantize(weight, sub_dim, n_centroids, iters=10, share_codebook=True,
            max_fit=None, channel_weight=None) -> tuple[Tensor, Tensor] # (codes, codebook)
pq_dequantize(codes, codebooks) -> Tensor
pq_bpw(out, in_, sub_dim, n_centroids, share_codebook=True) -> float

# src/smart_quant/encode.py
centroids_for_bits(bits, sub_dim, lo=16, hi=4096) -> int
quantize_fused_experts(weight, bits_per_expert, sub_dim, iters=10,
                       codebook_order=1, channel_weight=None) -> tuple[float, int]
quantize_experts(model, avg_bits, sub_dim=4, freqs=None, iters=10, bits_lo=1.5,
                 bits_hi=3.0, codebook_order=1, importance=None) -> list[dict]

# src/smart_quant/expert_importance.py
layer_index(name) -> int | None
ActivationImportanceProfiler(model, num_experts)   # __enter__/__exit__, .importance(granularity)

# tests/conftest.py
heterogeneous(*shape) -> Tensor        # columns with genuinely different scales
weighted_mse(recon, target, channel_weight) -> float
TinyExperts(num_experts=4, hidden=8, inter=4)      # fused (E, out, in) params + routing forward
wrap_in_layers(experts, prefix_depth=1)            # nests at ...layers.0.mlp.experts
```

---

### Task 1: `damped_inverse`

**Files:**
- Create: `src/smart_quant/compensate.py`
- Test: `tests/test_compensate.py` (new area, new file)

**Interfaces:**
- Consumes: nothing.
- Produces: `damped_inverse(h: torch.Tensor, damp: float = 0.01) -> torch.Tensor` — takes an
  `(n, n)` input covariance, returns the **upper** Cholesky factor of the damped `H⁻¹`, float64.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compensate.py`:

```python
import pytest
import torch

from smart_quant.compensate import damped_inverse


def correlated_inputs(n: int, dim: int, rank: int = 8, seed: int = 0) -> torch.Tensor:
    """(n, dim) activations with genuine off-diagonal structure: a low-rank factor plus noise.
    Uncorrelated inputs leave nothing for compensation to exploit, so fixtures must not use
    plain randn."""
    g = torch.Generator().manual_seed(seed)
    factor = torch.randn(dim, rank, generator=g)
    return torch.randn(n, rank, generator=g) @ factor.T + 0.3 * torch.randn(n, dim, generator=g)


class TestDampedInverse:
    def test_is_upper_triangular_cholesky_of_inverse(self):
        x = correlated_inputs(512, 16)
        h = x.T @ x / x.shape[0]
        u = damped_inverse(h, damp=0.01)
        assert torch.equal(u, torch.triu(u))
        # U^T U reconstructs the damped inverse
        damped = h.double() + 0.01 * torch.diag(h).double().mean() * torch.eye(16, dtype=torch.float64)
        assert torch.allclose(u.T @ u, torch.linalg.inv(damped), atol=1e-6)

    def test_survives_a_singular_covariance(self):
        # fewer samples than dimensions -> rank-deficient, exactly what cold experts produce
        x = correlated_inputs(8, 16)
        u = damped_inverse(x.T @ x / 8)
        assert torch.isfinite(u).all()

    def test_dead_channel_does_not_poison_the_factor(self):
        # an all-zero input channel gives a zero row/column; without handling, Cholesky fails
        x = correlated_inputs(512, 16)
        x[:, 5] = 0.0
        u = damped_inverse(x.T @ x / 512)
        assert torch.isfinite(u).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_compensate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'smart_quant.compensate'`

- [ ] **Step 3: Implement**

Create `src/smart_quant/compensate.py`:

```python
"""GPTQ-style off-diagonal error compensation for the shared-codebook product quantizer.

Every prior phase reweighted *what the fit optimizes*; the post-mortem in
`docs/weighting-diagnosis.md` closed that family — `E[x^2]` is the layerwise Hessian diagonal
and it is anti-correlated with where PQ actually errs. This module changes the algorithm
instead: quantize groups in order, and push each group's error onto the not-yet-quantized
columns along the directions the inputs are actually correlated in.

Nothing extra is stored. Compensation changes which codes get chosen, so the artifact is still
codes + codebook and `pq_bpw` is untouched.
"""
from __future__ import annotations

import torch

from smart_quant.codebook import assign, pq_quantize

__all__ = ["damped_inverse", "compensated_quantize"]


def damped_inverse(h: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor of the damped `H^-1`, in float64.

    `damp * mean(diag(H))` on the diagonal is GPTQ's standard preconditioning. It is
    load-bearing here, not decorative: layer 0's covariance has condition ~1e5 and any
    per-expert estimate over few routed tokens is outright rank-deficient. Dead input channels
    (all-zero column, which real experts have) get a unit diagonal so they contribute no
    compensation rather than producing a non-positive-definite matrix."""
    n = h.shape[0]
    h = h.double().clone()
    d = torch.arange(n, device=h.device)
    dead = torch.diagonal(h) == 0
    h[dead, dead] = 1.0
    h[d, d] += damp * torch.diagonal(h).mean()
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(hinv, upper=True)
```

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 60 passed (57 baseline + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/compensate.py tests/test_compensate.py
git commit -m "feat: damped Cholesky factor of the inverse input covariance"
```

---

### Task 2: `compensated_quantize` — single tensor

The correctness core. Task 3 adds the batched path for speed; this one is the reference.

**Files:**
- Modify: `src/smart_quant/compensate.py`
- Test: `tests/test_compensate.py`

**Interfaces:**
- Consumes: `damped_inverse` from Task 1; `assign`, `pq_quantize` from `codebook.py`.
- Produces:
  ```python
  compensated_quantize(
      weight: torch.Tensor,            # (out, in) float
      sub_dim: int,
      n_centroids: int,
      hinv_chol: torch.Tensor,         # (in, in) upper Cholesky from damped_inverse
      iters: int = 10,
      max_fit: int | None = None,
      rounds: int = 3,
      compensate: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor, list[float]]   # (codes, codebook, per_round_mse)
  ```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_compensate.py`:

```python
from smart_quant.codebook import pq_dequantize, pq_quantize
from smart_quant.compensate import compensated_quantize
from conftest import heterogeneous


def layer_error(w_hat: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> float:
    """||W X^T - West X^T||^2 — the objective GPTQ actually minimizes, not plain weight MSE."""
    return float(((w_hat - w) @ x.T).pow(2).mean())


class TestCompensatedQuantize:
    def test_lowers_the_layerwise_error_it_optimizes(self):
        torch.manual_seed(0)
        w = heterogeneous(64, 32)
        x = correlated_inputs(2048, 32, rank=6, seed=1)
        u = damped_inverse(x.T @ x / x.shape[0])
        plain = pq_dequantize(*pq_quantize(w, 4, 16, iters=10))
        codes, cb, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1)
        assert layer_error(pq_dequantize(codes, cb), w, x) < layer_error(plain, w, x)

    def test_diagonal_hessian_makes_compensation_a_near_no_op(self):
        # uncorrelated inputs leave nothing to push error onto; a version that "helps" here
        # is helping for the wrong reason
        torch.manual_seed(1)
        w = heterogeneous(64, 32)
        u = damped_inverse(torch.eye(32) * 2.0)
        on, _, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1, compensate=True)
        off, _, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1, compensate=False)
        # a diagonal H gives a diagonal Cholesky factor, so u[block, after] is exactly zero and
        # the update is a no-op — assert equality, not "close enough"
        assert torch.equal(on, off)

    def test_matches_per_column_gptq_at_sub_dim_1(self):
        # at sub_dim=1 the block update degenerates to textbook GPTQ; block formulations are
        # easy to get subtly wrong, so pin it against an explicit per-column reference
        torch.manual_seed(2)
        w = heterogeneous(16, 8)
        x = correlated_inputs(1024, 8, rank=3, seed=3)
        u = damped_inverse(x.T @ x / x.shape[0])
        codes, cb, _ = compensated_quantize(w, 1, 8, u, iters=20, rounds=1)

        work, ref = w.float().clone(), torch.empty_like(w)
        for j in range(8):
            idx = assign(work[:, j:j + 1], cb.float())
            ref[:, j] = cb.float()[idx].squeeze(1)
            err = work[:, j] - ref[:, j]
            if j + 1 < 8:
                work[:, j + 1:] -= torch.outer(err / u[j, j], u[j, j + 1:]).float()
        assert torch.allclose(pq_dequantize(codes, cb), ref, atol=1e-5)

    def test_compensate_false_is_byte_identical_to_pq_quantize(self):
        torch.manual_seed(3)
        w = heterogeneous(64, 32)
        u = damped_inverse(torch.eye(32))
        codes, cb, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1, compensate=False)
        base_codes, base_cb = pq_quantize(w, 4, 16, iters=10)
        assert torch.equal(codes, base_codes) and torch.equal(cb, base_cb)

    def test_reports_one_error_per_round_and_shapes_are_unchanged(self):
        torch.manual_seed(4)
        w = heterogeneous(64, 32)
        x = correlated_inputs(2048, 32, rank=6, seed=5)
        u = damped_inverse(x.T @ x / x.shape[0])
        codes, cb, errs = compensated_quantize(w, 4, 16, u, iters=10, rounds=3)
        base_codes, base_cb = pq_quantize(w, 4, 16, iters=10)
        assert len(errs) == 3
        assert codes.shape == base_codes.shape and cb.shape == base_cb.shape   # footprint
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_compensate.py::TestCompensatedQuantize -q`
Expected: FAIL — `ImportError: cannot import name 'compensated_quantize'`

- [ ] **Step 3: Implement**

Append to `src/smart_quant/compensate.py`:

```python
def compensated_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    n_centroids: int,
    hinv_chol: torch.Tensor,
    iters: int = 10,
    max_fit: int | None = None,
    rounds: int = 3,
    compensate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Quantize a (out, in) weight group-by-group, pushing each group's error onto the columns
    not yet quantized. Returns (codes, codebook, per-round reconstruction MSE).

    The quantization unit is a group of `sub_dim` adjacent input channels assigned atomically to
    one centroid, so per-column sequential quantization inside a group is unavailable — this is
    block-GPTQ with block = group. At `sub_dim=1` it reduces to textbook GPTQ.

    Each round refits the codebook on the *previous* round's compensated weights but replays the
    compensation pass from the original weights; compounding the correction would apply it
    repeatedly and diverge. Only `codes` and the final `codebook` are kept, so the footprint is
    identical to an uncompensated encode."""
    out, in_ = weight.shape
    if in_ % sub_dim:
        raise ValueError(f"in_features {in_} not divisible by sub_dim {sub_dim}")
    groups = in_ // sub_dim
    u = hinv_chol.to(device=weight.device, dtype=torch.float32)
    w_fit, errors = weight, []
    codes = torch.empty(out, groups, dtype=torch.long, device=weight.device)

    for _ in range(rounds):
        codebook = pq_quantize(w_fit, sub_dim, n_centroids, iters, max_fit=max_fit)[1].float()
        work = weight.float().clone()
        for g in range(groups):
            lo, hi = g * sub_dim, (g + 1) * sub_dim
            codes[:, g] = assign(work[:, lo:hi], codebook)
            if compensate and hi < in_:
                err = work[:, lo:hi] - codebook[codes[:, g]]
                delta = err @ torch.linalg.inv(u[lo:hi, lo:hi])
                work[:, hi:] -= delta @ u[lo:hi, hi:]
        errors.append(float((codebook[codes].reshape(out, in_) - weight).pow(2).mean()))
        w_fit = work
    return codes, codebook.to(weight.dtype), errors
```

The per-round error is plain reconstruction MSE against the original weights — cheap, and enough
to answer "does round 3 add anything over round 2". The layerwise objective needs `X`, which the
encode does not carry.

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 65 passed (60 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/compensate.py tests/test_compensate.py
git commit -m "feat: block-GPTQ error compensation over codebook groups"
```

---

### Task 3: Batched path over experts

Without this the encode is 5.2M Python iterations (512 groups × 256 experts × 40 layers) and is
kernel-launch bound — the same failure that killed the Phase-7 profiler's first draft. All experts
in a layer share `H` and the group order, so the loop runs once and the work batches.

**Files:**
- Modify: `src/smart_quant/compensate.py`
- Test: `tests/test_compensate.py`

**Interfaces:**
- Consumes: `compensated_quantize` (Task 2) as the correctness reference.
- Produces:
  ```python
  compensated_quantize_fused(
      weight: torch.Tensor,            # (n_experts, out, in) — modified in place
      sub_dim: int,
      n_centroids: int,
      hinv_chol: torch.Tensor,
      iters: int = 10,
      max_fit: int | None = None,
      rounds: int = 3,
      compensate: bool = True,
  ) -> list[float]                     # per-round mean MSE across experts
  ```
  Writes the dequantized reconstruction back into `weight`, matching `quantize_fused_experts`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_compensate.py`:

```python
from smart_quant.compensate import compensated_quantize_fused


class TestBatchedCompensation:
    def test_matches_the_single_tensor_reference_expert_for_expert(self):
        torch.manual_seed(0)
        base = torch.stack([heterogeneous(32, 16) for _ in range(3)])
        x = correlated_inputs(1024, 16, rank=4, seed=7)
        u = damped_inverse(x.T @ x / x.shape[0])

        batched = base.clone()
        compensated_quantize_fused(batched, 4, 16, u, iters=10, rounds=2)
        for e in range(3):
            codes, cb, _ = compensated_quantize(base[e], 4, 16, u, iters=10, rounds=2)
            assert torch.allclose(batched[e], pq_dequantize(codes, cb), atol=1e-5)

    def test_writes_reconstruction_in_place_and_reports_per_round(self):
        torch.manual_seed(1)
        w = torch.stack([heterogeneous(32, 16) for _ in range(2)])
        orig = w.clone()
        errs = compensated_quantize_fused(w, 4, 16, damped_inverse(torch.eye(16)), rounds=3)
        assert len(errs) == 3
        assert not torch.equal(w, orig) and torch.isfinite(w).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_compensate.py::TestBatchedCompensation -q`
Expected: FAIL — `ImportError: cannot import name 'compensated_quantize_fused'`

- [ ] **Step 3: Implement**

Append to `src/smart_quant/compensate.py`:

```python
def compensated_quantize_fused(
    weight: torch.Tensor,
    sub_dim: int,
    n_centroids: int,
    hinv_chol: torch.Tensor,
    iters: int = 10,
    max_fit: int | None = None,
    rounds: int = 3,
    compensate: bool = True,
) -> list[float]:
    """`compensated_quantize` over a fused (n_experts, out, in) tensor, writing the
    reconstruction back in place. Every expert in a layer shares `H` and the group order, so the
    group loop runs once and assignment/compensation batch across experts — ~20k batched steps
    for the whole model instead of 5.2M Python iterations.

    Codebooks stay per-expert (as elsewhere in this codebase), so assignment is a batched cdist
    against each expert's own centroids."""
    n_experts, out, in_ = weight.shape
    groups = in_ // sub_dim
    u = hinv_chol.to(device=weight.device, dtype=torch.float32)
    original = weight.float().clone()
    w_fit, errors = original, []
    codes = torch.empty(n_experts, out, groups, dtype=torch.long, device=weight.device)

    for _ in range(rounds):
        books = torch.stack([
            pq_quantize(w_fit[e], sub_dim, n_centroids, iters, max_fit=max_fit)[1].float()
            for e in range(n_experts)
        ])                                                   # (n_experts, k, sub_dim)
        work = original.clone()
        for g in range(groups):
            lo, hi = g * sub_dim, (g + 1) * sub_dim
            codes[:, :, g] = torch.cdist(work[:, :, lo:hi], books).argmin(dim=2)
            if compensate and hi < in_:
                block_recon = torch.gather(
                    books, 1, codes[:, :, g].unsqueeze(-1).expand(-1, -1, sub_dim))
                delta = (work[:, :, lo:hi] - block_recon) @ torch.linalg.inv(u[lo:hi, lo:hi])
                work[:, :, hi:] -= delta @ u[lo:hi, hi:]
        full_recon = torch.gather(
            books, 1, codes.reshape(n_experts, -1, 1).expand(-1, -1, sub_dim)
        ).reshape(n_experts, out, in_)
        errors.append(float((full_recon - original).pow(2).mean()))
        w_fit = work

    weight.copy_(full_recon.to(weight.dtype))
    return errors
```

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 67 passed (65 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/compensate.py tests/test_compensate.py
git commit -m "feat: batched compensation across experts sharing a layer Hessian"
```

---

### Task 4: `HessianProfiler`

**Files:**
- Modify: `src/smart_quant/expert_importance.py`
- Test: `tests/test_expert_importance.py`

**Interfaces:**
- Consumes: `layer_index` (already in this module).
- Produces: `HessianProfiler(model: nn.Module)` with `__enter__`/`__exit__` and
  `.hessians() -> dict[int, torch.Tensor]` keyed by **layer index**, each `(in, in)` fp32 on CPU.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_expert_importance.py`:

```python
class TestHessianProfiler:
    def test_accumulates_the_input_second_moment_per_layer(self):
        torch.manual_seed(0)
        experts = TinyExperts()
        model = wrap_in_layers(experts, prefix_depth=1)
        x = torch.randn(12, 8)
        with HessianProfiler(model) as prof:
            model.inner.layers[0].mlp.experts(
                x, torch.zeros(12, 1, dtype=torch.long), torch.ones(12, 1))
        h = prof.hessians()[0]
        assert h.shape == (8, 8)
        assert torch.allclose(h, x.T @ x / 12, atol=1e-5)

    def test_keys_are_layer_indices_not_module_paths(self):
        model = wrap_in_layers(TinyExperts(), prefix_depth=2)
        with HessianProfiler(model) as prof:
            model.inner.inner.layers[0].mlp.experts(
                torch.randn(4, 8), torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1))
        assert list(prof.hessians()) == [0]

    def test_hooks_removed_on_exit(self):
        experts = TinyExperts()
        with HessianProfiler(wrap_in_layers(experts)) as prof:
            pass
        assert prof.handles == [] and not experts._forward_pre_hooks
```

Add `HessianProfiler` to the import line at the top of the file.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_expert_importance.py::TestHessianProfiler -q`
Expected: FAIL — `ImportError: cannot import name 'HessianProfiler'`

- [ ] **Step 3: Implement**

Append to `src/smart_quant/expert_importance.py`:

```python
class HessianProfiler:
    """Accumulates the input second moment `X^T X` per MoE layer over a calibration set.

    Keyed by layer *index*, never module path, so the artifact survives the different wrapper
    prefixes `AutoModel` (the profiler) and `AutoModelForCausalLM` (the encode) put on the same
    modules — the failure mode that nearly shipped in Phase 7.

    This is `gate_up_proj`'s input covariance: the hidden states entering the block, shared by
    every expert in the layer. `down_proj`'s input is the expert-specific intermediate and has no
    such shortcut; Phase 8 leaves it uniform.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.acc: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(self, layer: int):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            gram = x.T @ x
            self.acc[layer] = gram if layer not in self.acc else self.acc[layer] + gram
            self.counts[layer] = self.counts.get(layer, 0) + x.shape[0]
        return hook

    def __enter__(self) -> "HessianProfiler":
        for name, module in self.model.named_modules():
            layer = layer_index(name)
            if type(module).__name__.endswith("Experts") and layer is not None:
                self.handles.append(module.register_forward_pre_hook(self.make_hook(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def hessians(self) -> dict[int, torch.Tensor]:
        """Mean `x x^T` per layer, on CPU."""
        return {k: (v / max(self.counts[k], 1)).cpu() for k, v in self.acc.items()}
```

Add `"HessianProfiler"` to the module's `__all__` if one is present.

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 70 passed (67 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/expert_importance.py tests/test_expert_importance.py
git commit -m "feat: per-layer input-covariance profiler for compensation"
```

---

### Task 5: Encode-path wiring

**Files:**
- Modify: `src/smart_quant/encode.py`
- Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: `compensated_quantize_fused` (Task 3), `damped_inverse` (Task 1).
- Produces:
  - `quantize_fused_experts(..., hinv_chol: torch.Tensor | None = None, rounds: int = 3, compensate: bool = True)`
  - `quantize_experts(..., hessians: dict[int, torch.Tensor] | None = None, rounds: int = 3, compensate: bool = True)`

  Compensation applies to `gate_up_proj` **only**; `down_proj` takes the existing uniform path and
  is the phase's control.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_encode.py`:

```python
from smart_quant.compensate import damped_inverse
from smart_quant.expert_importance import HessianProfiler


class TestCompensatedEncode:
    def test_absent_hessian_is_byte_identical(self):
        torch.manual_seed(0)
        base = heterogeneous(2, 32, 16)
        a, b = base.clone(), base.clone()
        r1 = quantize_fused_experts(a, torch.full((2,), 2.0), sub_dim=4, iters=5)
        r2 = quantize_fused_experts(b, torch.full((2,), 2.0), sub_dim=4, iters=5, hinv_chol=None)
        assert r1 == r2 and torch.equal(a, b)

    def test_footprint_is_unchanged_by_compensation(self):
        # compensation moves codes, never bit counts — the matched-comparison guarantee
        torch.manual_seed(1)
        w = heterogeneous(2, 32, 16)
        u = damped_inverse(torch.eye(16) * 2)
        plain = quantize_fused_experts(w.clone(), torch.full((2,), 2.0), sub_dim=4, iters=5)
        comp = quantize_fused_experts(w.clone(), torch.full((2,), 2.0), sub_dim=4, iters=5,
                                      hinv_chol=u, rounds=2)
        assert plain == comp

    def test_only_gate_up_is_compensated(self):
        # down_proj is the control: it must come out identical with and without a Hessian
        torch.manual_seed(2)
        experts = TinyExperts()
        model = wrap_in_layers(copy.deepcopy(experts), prefix_depth=1)
        with HessianProfiler(model) as prof:
            model.inner.layers[0].mlp.experts(
                torch.randn(64, 8), torch.zeros(64, 1, dtype=torch.long), torch.ones(64, 1))
        hess = prof.hessians()

        plain = wrap_in_layers(copy.deepcopy(experts), prefix_depth=1)
        comp = wrap_in_layers(copy.deepcopy(experts), prefix_depth=1)
        quantize_experts(plain, avg_bits=2.0, iters=3)
        quantize_experts(comp, avg_bits=2.0, iters=3, hessians=hess, rounds=2)
        pe = plain.inner.layers[0].mlp.experts
        ce = comp.inner.layers[0].mlp.experts
        assert torch.equal(pe.down_proj, ce.down_proj)          # control untouched
        assert not torch.equal(pe.gate_up_proj, ce.gate_up_proj)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_encode.py::TestCompensatedEncode -q`
Expected: FAIL — `TypeError: quantize_fused_experts() got an unexpected keyword argument 'hinv_chol'`

- [ ] **Step 3: Implement `quantize_fused_experts`**

Add the parameters and dispatch. When a Hessian factor is supplied the batched compensated path
replaces the per-expert loop entirely:

```python
def quantize_fused_experts(
    weight: torch.Tensor, bits_per_expert: torch.Tensor, sub_dim: int, iters: int = 10,
    codebook_order: int = 1, channel_weight: torch.Tensor | None = None,
    hinv_chol: torch.Tensor | None = None, rounds: int = 3, compensate: bool = True,
) -> tuple[float, int]:
    """... (existing docstring) ...

    `hinv_chol` enables GPTQ-style error compensation across this tensor's input dim. It changes
    which codes are chosen, never the bit count, so the returned footprint is identical to the
    uncompensated encode. Requires `codebook_order == 1` and no `channel_weight` — compensation
    and the Phase-7 weighting are separate techniques and are not combined."""
    if hinv_chol is not None:
        if codebook_order != 1 or channel_weight is not None:
            raise ValueError("compensation supports codebook_order=1 without channel_weight")
        k = centroids_for_bits(float(bits_per_expert[0]), sub_dim)
        compensated_quantize_fused(weight, sub_dim, k, hinv_chol, iters=iters,
                                   max_fit=max(4096, k * 8), rounds=rounds,
                                   compensate=compensate)
        out, in_ = weight.shape[1], weight.shape[2]
        n_experts = weight.shape[0]
        return pq_bpw(out, in_, sub_dim, [k]) * out * in_ * n_experts, out * in_ * n_experts

    # ... existing per-expert loop unchanged ...
```

Note the compensated path uses a single `k` for all experts: batching over experts requires one
codebook size, so per-expert bit allocation (`freqs`) and compensation are mutually exclusive.
Every Phase-8 encode is uniform-allocation, so this costs nothing here.

Import at the top: `from smart_quant.compensate import compensated_quantize_fused`.

- [ ] **Step 4: Implement `quantize_experts`**

Thread the Hessian through, `gate_up_proj` only:

```python
def quantize_experts(
    model, avg_bits: float, sub_dim: int = 4, freqs: dict | None = None, iters: int = 10,
    bits_lo: float = 1.5, bits_hi: float = 3.0, codebook_order: int = 1,
    importance: dict[str, torch.Tensor] | None = None,
    hessians: dict[int, torch.Tensor] | None = None, rounds: int = 3,
    compensate: bool = True,
) -> list[dict]:
    """... (existing docstring) ...

    `hessians` maps layer index to that layer's input second moment. Compensation is applied to
    `gate_up_proj` only — `down_proj`'s intermediate is expert-specific and nearly isotropic, so
    it stays uniform and serves as the phase's control."""
```

Inside the per-parameter loop, alongside the existing `cw` lookup:

```python
                hin = None
                if hessians is not None and param_name == "gate_up_proj":
                    h = hessians.get(layer_index(name))
                    if h is not None:
                        hin = damped_inverse(h)
                        matched += 1
                fused_bits, fused_weights = quantize_fused_experts(
                    weight, bits, sub_dim, iters, codebook_order=codebook_order,
                    channel_weight=cw, hinv_chol=hin, rounds=rounds, compensate=compensate)
```

Extend the existing unmatched-artifact guard so a supplied `hessians` that matches nothing raises
rather than silently producing an uncompensated encode labelled as compensated — the same failure
the Phase-7 review caught:

```python
    if (importance is not None or hessians is not None) and not matched:
        raise KeyError(
            "importance/hessians supplied but no key matched any fused expert parameter. "
            "Expected importance keys '<layer_index>.<param_name>' and hessian keys <layer_index>.")
```

Import `damped_inverse` at the top of `encode.py`.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 73 passed (70 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/smart_quant/encode.py tests/test_encode.py
git commit -m "feat: wire error compensation into the encode path for gate_up_proj"
```

---

### Task 6: CLI

**Files:**
- Modify: `src/smart_quant/cli.py`

- [ ] **Step 1: Add `profile-hessian`**

Append before `if __name__ == "__main__":`, reusing the shared calibration helpers added in Phase 7:

```python
@app.command("profile-hessian")
def profile_hessian(
    model: str = typer.Option(..., help="HF repo id or local path."),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    out: Path = typer.Option(Path("experiments/expert_hessian.pt")),
) -> None:
    """Accumulate the per-layer input second moment for error compensation."""
    import torch

    from smart_quant.expert_importance import HessianProfiler

    tok, lm, _, rows = load_for_calibration(model)
    with HessianProfiler(lm) as prof:
        stream_calibration(tok, lm, rows, calib_rows, seq_len)
        hess = prof.hessians()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hess, out)
    console.print(f"profiled {len(hess)} layer Hessians over {calib_rows} rows → {out}")
```

- [ ] **Step 2: Add the `encode-eval` options**

Add to the `encode_eval` signature, after `importance_path`:

```python
    hessian_path: Path | None = typer.Option(
        None, help="Per-layer Hessian .pt from profile-hessian; enables compensation."),
    rounds: int = typer.Option(3, help="Fit/compensate rounds."),
    compensate: bool = typer.Option(True, help="--no-compensate runs the refit-only control."),
```

Load and pass, next to the existing `importance` load:

```python
    hessians = torch.load(hessian_path, weights_only=True) if hessian_path else None
    stats = quantize_experts(lm, avg_bits=avg_bits, sub_dim=sub_dim, freqs=freqs,
                             bits_lo=bits_lo, bits_hi=bits_hi, codebook_order=codebook_order,
                             importance=importance, hessians=hessians, rounds=rounds,
                             compensate=compensate)
```

Record it on the results row so `results.jsonl` is self-describing — derived from what was
actually loaded, never a hand-typed flag (the Phase-7 lesson):

```python
           "compensation": None if hessians is None else (
               f"rounds={rounds}" if compensate else f"refit-only rounds={rounds}"),
```

- [ ] **Step 3: Verify the CLI loads**

Run:
```bash
PYTHONPATH=src .venv/bin/python -c "
import inspect
from smart_quant import cli
print('profile_hessian:', 'profile_hessian' in dir(cli))
print('encode_eval:', [p for p in inspect.signature(cli.encode_eval).parameters
                       if p in ('hessian_path', 'rounds', 'compensate')])"
```
Expected: `profile_hessian: True` and all three parameter names listed.

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 73 passed, unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/cli.py
git commit -m "feat: profile-hessian command and compensation options on encode-eval"
```

---

### Task 7: Calibration and encodes on the box

No TDD cycle — this is the experiment. Sequential; only one fp16 model fits in 80 GB.

- [ ] **Step 1: Push and sync the box**

```bash
git push -u origin feat/gptq-compensation-phase8
ssh pi-a100-80gb 'cd ~/small-smart-models && git fetch origin && \
  git checkout feat/gptq-compensation-phase8 && \
  PYTHONPATH=src .venv/bin/python -m pytest -q | tail -2'
```
Expected: 73 passed on the box.

- [ ] **Step 2: Smoke-test the Hessian pass before committing to the full run**

The box venv has no installed package — `PYTHONPATH=src` is required, and a 2-row smoke catches
that in seconds rather than after a 2-minute model load inside a detached run (Phase-7 lesson):

```bash
ssh pi-a100-80gb 'cd ~/small-smart-models && PYTHONPATH=src .venv/bin/python -m smart_quant.cli \
  profile-hessian --model Qwen/Qwen3.6-35B-A3B --calib-rows 2 --seq-len 512 \
  --out /tmp/smoke_hess.pt 2>&1 | tail -3'
```
Expected: `profiled 40 layer Hessians over 2 rows`. Then verify shape and keys:

```bash
ssh pi-a100-80gb 'cd ~/small-smart-models && PYTHONPATH=src .venv/bin/python -c "
import torch; h = torch.load(\"/tmp/smoke_hess.pt\", weights_only=True)
print(sorted(h)[:3], tuple(h[0].shape), bool(torch.isfinite(h[0]).all()))"'
```
Expected: `[0, 1, 2] (2048, 2048) True`.

- [ ] **Step 3: Write the sequential runner**

Create `run_gptq_seq.sh` on the box (not committed, matching `run_rvq_seq.sh` / `run_wpq_seq.sh`):

```bash
export PYTHONPATH=src
PY=.venv/bin/python
M=Qwen/Qwen3.6-35B-A3B
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[seq] $(ts) START hessian calibration"
$PY -m smart_quant.cli profile-hessian --model "$M" --out experiments/expert_hessian.pt
rc=$?
echo "[seq] $(ts) DONE hessian exit=$rc"
[ $rc -ne 0 ] && { echo "[seq] $(ts) ABORT"; exit 1; }

run() {  # label avg_bits extra_flags
  echo "[seq] $(ts) START $1"
  $PY -m smart_quant.cli encode-eval --model "$M" --label "$1" --avg-bits "$2" \
    --hessian-path experiments/expert_hessian.pt --rounds 3 $3
  echo "[seq] $(ts) DONE $1 exit=$?"
}

run refit25-control 2.5 --no-compensate
run gptq25 2.5
run gptq20 2.0
echo "[seq] $(ts) ALL-DONE"
```

- [ ] **Step 4: Launch detached**

```bash
ssh pi-a100-80gb 'cd ~/small-smart-models && setsid nohup bash run_gptq_seq.sh </dev/null \
  >>logs/gptq_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & exit 0'
ssh pi-a100-80gb 'ps -o pid,ppid,sid,cmd -C bash | grep gptq_seq'
```

Confirm `SID == PID` (own session, SIGHUP-immune). The parent may briefly be the ssh wrapper; that
wrapper is itself init-owned and exits shortly.

- [ ] **Step 5: Gate on the footprint before the remaining encodes**

As soon as `refit25-control` appends:

```bash
ssh pi-a100-80gb 'cd ~/small-smart-models && grep -E "\"label\": \"(pq25-uniform|refit25-control)\"" \
  experiments/bits-per-brain/results.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    r = json.loads(l); print(r[\"label\"], r[\"expert_bpw\"], r.get(\"compensation\"))"'
```

Expected: **both exactly `2.542`.** Compensation and refitting change which codes are chosen and
nothing else, so any footprint difference at all means the accounting moved — stop and diagnose
rather than reporting a comparison that is no longer matched.

- [ ] **Step 6: Regenerate the plot and commit the PNG**

`gptq*` and `refit*` labels start with neither `wpq` nor `rvq` and do not end in `-uniform`, so they
are invisible to all three existing curve selectors. Add a `Family("gptq", ...)` entry to `FAMILIES`
in [`experiments/plot_quality_vs_bpw.py`](../../experiments/plot_quality_vs_bpw.py) — one row, no
other change — then:

```bash
PYTHONPATH=src .venv/bin/python experiments/plot_quality_vs_bpw.py
git add experiments/plot_quality_vs_bpw.py experiments/progress/bits-per-brain/quality-vs-bpw.png
git commit -m "feat: GPTQ curve on the quality-vs-bpw plot"
```

---

### Task 8: Write up and open the PR

- [ ] **Step 1: Add the Phase-8 section to `docs/experiments/bits-per-brain.md`**

After the Phase-7 section, same shape (hypothesis, matched-footprint table, verdict, design link):

```markdown
### Phase 8 — GPTQ-style off-diagonal error compensation

| footprint | uniform PQ | refit-only control | compensated |
|---|---|---|---|
| ~2.0 bpw | `pq2` 6.765 | — | `gptq20` _ppl_ |
| ~2.5 bpw | `pq25` (2.542) 6.2137 | `refit25-control` (2.542) _ppl_ | `gptq25` (2.542) _ppl_ |
```

State whether `gptq25` beat **both** the baseline and the control — beating only the baseline means
refitting did the work. Report the per-round reconstruction errors: if round 3 did not improve on
round 2, say so, since that is the drift question answered by measurement. Record that `down_proj`
was left uniform, so a gain reaches perplexity at roughly ⅔ strength.

- [ ] **Step 2: Run the suite one final time**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 73 passed.

- [ ] **Step 3: Simplify pass before pushing**

Run `/simplify` on the working diff and fold findings into the commits they belong to — **before**
pushing, not after. In Phase 7 this pass caught an experiment-invalidating key-mismatch bug; the
compensation path has the same producer/consumer artifact shape, so it is worth the same scrutiny.

- [ ] **Step 4: Commit, push, open the PR**

```bash
git add docs/experiments/bits-per-brain.md
git commit -m "docs: Phase-8 GPTQ error-compensation results"
git push origin feat/gptq-compensation-phase8
```

Then `gh pr create` with the single-quoted heredoc form, standard section order (Summary / Test plan
/ Visual aid / Commits / Out-of-scope follow-ups), grouped commit tables, and every symbol
deep-linked to `../tree/feat/gptq-compensation-phase8/<path>`. Render-check with
`gh pr view <N> --json body --jq '.body' | head -40`.

## Out of scope

- **Per-expert Hessians** — the refinement if per-layer shows signal; gated on the cold-expert sampling problem.
- **`down_proj` compensation** — near-isotropic inputs, and it is the control here.
- **Activation reordering (`act order`)** — GPTQ's descending-`diag(H)` group order; deferred so this phase tests one thing.
- **End-loss Fisher** — the other surviving direction from the post-mortem; needs backprop.
- **`.pre-commit-config.yaml` ruff C901/PLR0915 gate** — repo-wide follow-up, still absent.
