# Phase 7 — Activation-Weighted First-Order PQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add activation-derived importance weighting to the first-order product quantizer and measure whether it beats uniform PQ at matched footprint.

**Architecture:** Importance is per input channel, which is axis 0 of each `weight[e]` slice — and PQ splits sub-vectors along axis 1. So every element of a sub-vector shares one scalar weight, and the whole feature reduces to a `sample_weight` argument on `lloyd_kmeans` that changes only the centroid update. `pq_dequantize` and `pq_bpw` are untouched, so realized bpw is identical to the unweighted encode and the matched comparison holds by construction.

**Tech Stack:** Python 3.13, PyTorch (CPU for unit tests, CUDA on the box), typer CLI, pytest, matplotlib for the plot.

**Spec:** [`docs/specs/2026-07-29-weighted-pq-phase7-design.md`](../specs/2026-07-29-weighted-pq-phase7-design.md)

## Global Constraints

- Branch `feat/weighted-pq-phase7`, off `main`. Never commit to `main`; the phase lands as one PR.
- Conventional commits (`feat:`, `test:`, `docs:`). No `Co-Authored-By` trailers.
- Type hints on every new signature, params **and** return, with explicit generics: `dict[str, torch.Tensor]`, never bare `dict`.
- No leading underscores on module-level functions, classes, or constants.
- Tests go in the **existing** files by area — `tests/test_codebook.py`, `tests/test_encode.py`, `tests/test_expert_importance.py`. Do not create new test modules.
- The CI surface is torch-only (`pyproject.toml` `[dependency-groups] test`). No test may import `transformers` or `datasets`.
- Every default must preserve existing behaviour: absent weights take the current code path and Phase-5/6 encodes stay byte-identical.
- Run the full suite with `PYTHONPATH=src .venv/bin/python -m pytest -q` before each commit. Baseline is **35 passed**.

---

### Task 1: Box reconnaissance — verify the `Experts` module tree

Gates Task 5's hook design. Independent of Tasks 2–4, so it can run in parallel with them, but must complete before Task 5 starts. No code ships from this task — the deliverable is a findings note committed to the repo.

**Files:**
- Create: `docs/plans/2026-07-29-phase7-recon-findings.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the findings note, which Task 5 reads to decide between the full design and the `gate_up_proj`-only fallback.

- [ ] **Step 1: Dump the expert module tree on the box**

```bash
ssh pi-a100-80gb
cd ~/small-smart-models
PYTHONPATH=src .venv/bin/python - <<'PY'
import inspect, torch
from transformers import AutoModel
lm = AutoModel.from_pretrained("Qwen/Qwen3.6-35B-A3B", torch_dtype="auto", device_map="meta")
for name, mod in lm.named_modules():
    if type(mod).__name__.endswith("Experts"):
        print("MODULE:", name, type(mod).__name__)
        for pn, p in mod.named_parameters(recurse=False):
            print("   param:", pn, tuple(p.shape), "dim", p.dim())
        print(inspect.signature(mod.forward))
        print(inspect.getsource(type(mod).forward))
        break
PY
```

- [ ] **Step 2: Record the four answers the profiler needs**

Write `docs/plans/2026-07-29-phase7-recon-findings.md` answering exactly these, with the pasted source as evidence:

1. Exact fused parameter names and shapes (expected `gate_up_proj` `(E, hidden, 2*inter)`, `down_proj` `(E, inter, hidden)`).
2. Does `forward` receive routing indices as an argument? If yes, the profiler can read them directly instead of pairing with a router hook.
3. Is the post-activation intermediate exposed anywhere, or purely internal?
4. How are gate and up packed in `gate_up_proj` — `chunk(2, dim=-1)`, interleaved, or otherwise? This determines whether the intermediate can be recomputed.

- [ ] **Step 3: State the verdict explicitly**

End the note with one of:
- **FULL** — intermediate recomputable, both projections get real statistics.
- **FALLBACK** — gate/up packing not cleanly splittable; weight `gate_up_proj` only, hold `down_proj` uniform, and record it as a stated limitation of the phase.

- [ ] **Step 4: Commit**

```bash
git add docs/plans/2026-07-29-phase7-recon-findings.md
git commit -m "docs: Phase-7 recon findings on Qwen3.6 Experts module tree"
```

---

### Task 2: Weighted `lloyd_kmeans`

**Files:**
- Modify: `src/smart_quant/codebook.py:21-35`
- Test: `tests/test_codebook.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `lloyd_kmeans(x: torch.Tensor, k: int, iters: int = 10, sample_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]`, returning `(centroids (k, d), assignment (n,))`. `sample_weight` has shape `(n,)` and is normalized to mean 1 internally.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_codebook.py`:

```python
class TestWeightedKMeans:
    @pytest.mark.parametrize("weights", [None, "ones"])
    def test_unweighted_paths_are_identical(self, weights):
        torch.manual_seed(0)
        x = torch.randn(512, 4)
        w = None if weights is None else torch.ones(512)
        base_c, base_i = lloyd_kmeans(x, k=16, iters=8)
        c, i = lloyd_kmeans(x, k=16, iters=8, sample_weight=w)
        assert torch.equal(c, base_c)
        assert torch.equal(i, base_i)

    def test_centroid_pulled_toward_heavy_cluster(self):
        # two tight clusters at -1 and +1; weighting the +1 cluster 10x must drag the
        # single centroid's weighted mean toward it
        torch.manual_seed(1)
        lo = torch.full((100, 2), -1.0) + 0.01 * torch.randn(100, 2)
        hi = torch.full((100, 2), 1.0) + 0.01 * torch.randn(100, 2)
        x = torch.cat([lo, hi])
        w = torch.cat([torch.ones(100), torch.full((100,), 10.0)])
        unweighted = lloyd_kmeans(x, k=1, iters=10)[0]
        weighted = lloyd_kmeans(x, k=1, iters=10, sample_weight=w)[0]
        assert unweighted.mean().abs() < 0.05          # balanced -> near origin
        assert weighted.mean() > 0.7                   # dragged toward +1

    def test_scale_invariant_in_weights(self):
        torch.manual_seed(2)
        x = torch.randn(256, 4)
        w = torch.rand(256) + 0.1
        a = lloyd_kmeans(x, k=8, iters=6, sample_weight=w)[0]
        b = lloyd_kmeans(x, k=8, iters=6, sample_weight=w * 1000.0)[0]
        assert torch.allclose(a, b, atol=1e-5)
```

Update the import at the top of the file to include `lloyd_kmeans`:

```python
from smart_quant.codebook import (
    lloyd_kmeans, pq_bpw, pq_dequantize, pq_quantize, residual_pq_quantize)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_codebook.py::TestWeightedKMeans -v`
Expected: FAIL — `TypeError: lloyd_kmeans() got an unexpected keyword argument 'sample_weight'`

- [ ] **Step 3: Implement**

Replace `lloyd_kmeans` in `src/smart_quant/codebook.py`:

```python
def lloyd_kmeans(x: torch.Tensor, k: int, iters: int = 10,
                 sample_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's k-means over rows of x (n, d), deterministic linspace init. Returns
    (centroids (k, d), assignment (n,)). Empty clusters keep their previous centroid.
    Centroid update is vectorized (index_add) — no Python loop over k.

    `sample_weight` (n,) weights the centroid *update* only — assignment is nearest-centroid
    regardless of weight. Weights are normalized to mean 1, so the fit is invariant to their
    scale. None reproduces the unweighted fit exactly."""
    n = x.shape[0]
    centroids = x[torch.linspace(0, n - 1, k).round().long()].clone()
    if sample_weight is None:
        w = torch.ones(n, device=x.device, dtype=x.dtype)
    else:
        w = sample_weight.to(device=x.device, dtype=x.dtype)
        w = w * (n / w.sum().clamp(min=1e-9))
    for _ in range(iters):
        idx = torch.cdist(x, centroids).argmin(dim=1)
        sums = torch.zeros_like(centroids).index_add_(0, idx, w.unsqueeze(1) * x)
        counts = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(0, idx, w)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    idx = torch.cdist(x, centroids).argmin(dim=1)
    return centroids, idx
```

The `None` path stays byte-identical because `1.0 * x` is exact in floating point and integer counts make `counts > 0` the same mask as before.

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 39 passed (35 baseline + 4 new cases; the parametrized identity test counts as 2).

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/codebook.py tests/test_codebook.py
git commit -m "feat: optional sample_weight on lloyd_kmeans"
```

---

### Task 3: Thread `sample_weight` through `pq_quantize` and `residual_pq_quantize`

**Files:**
- Modify: `src/smart_quant/codebook.py:38-73` (`pq_quantize`), `src/smart_quant/codebook.py:91-112` (`residual_pq_quantize`)
- Test: `tests/test_codebook.py`

**Interfaces:**
- Consumes: `lloyd_kmeans(..., sample_weight=...)` from Task 2.
- Produces:
  - `pq_quantize(weight, sub_dim, n_centroids, iters=10, share_codebook=True, max_fit=None, sample_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]` — `sample_weight` has shape `(out,)`, one per input channel.
  - `residual_pq_quantize(weight, sub_dim, stage_centroids, iters=10, share_codebook=True, max_fit=None, sample_weight: torch.Tensor | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_codebook.py`:

```python
class TestWeightedProductQuantization:
    def test_none_weight_matches_unweighted(self):
        torch.manual_seed(0)
        w = torch.randn(128, 32)
        a = pq_quantize(w, sub_dim=4, n_centroids=64, sample_weight=None)
        b = pq_quantize(w, sub_dim=4, n_centroids=64)
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_heavy_rows_reconstruct_better(self):
        # weighting rows 0:16 heavily must lower THEIR error relative to the unweighted fit,
        # which is the entire point of the feature
        torch.manual_seed(4)
        w = torch.randn(128, 32)
        sw = torch.ones(128)
        sw[:16] = 50.0
        plain = pq_dequantize(*pq_quantize(w, 4, 16, iters=15))
        weighted = pq_dequantize(*pq_quantize(w, 4, 16, iters=15, sample_weight=sw))
        err_plain = (plain[:16] - w[:16]).pow(2).mean()
        err_weighted = (weighted[:16] - w[:16]).pow(2).mean()
        assert err_weighted < err_plain

    def test_max_fit_keeps_weights_aligned_to_points(self):
        # Weight only the rows that survive an even subsample. If weights were applied to the
        # unsubsampled pool (or not subsampled at all), the fit would see a different weighting
        # and the heavy rows' error would not improve.
        torch.manual_seed(5)
        w = torch.randn(256, 32)
        sw = torch.zeros(256)
        sw[::2] = 1.0
        codes, cb = pq_quantize(w, 4, 16, iters=15, max_fit=512, sample_weight=sw)
        recon = pq_dequantize(codes, cb)
        err_weighted_rows = (recon[::2] - w[::2]).pow(2).mean()
        err_ignored_rows = (recon[1::2] - w[1::2]).pow(2).mean()
        assert err_weighted_rows < err_ignored_rows

    def test_pergroup_path_accepts_weights(self):
        torch.manual_seed(6)
        w = torch.randn(64, 32)
        codes, cb = pq_quantize(w, 8, 16, share_codebook=False, sample_weight=torch.rand(64) + 0.1)
        assert codes.shape == (64, 4)
        assert cb.shape == (4, 16, 8)

    def test_residual_forwards_weight_to_every_stage(self):
        torch.manual_seed(7)
        w = torch.randn(128, 32)
        sw = torch.ones(128)
        sw[:16] = 50.0
        plain = residual_pq_quantize(w, 4, [16, 16])
        weighted = residual_pq_quantize(w, 4, [16, 16], sample_weight=sw)
        err_plain = (pq_dequantize(*plain)[:16] - w[:16]).pow(2).mean()
        err_weighted = (pq_dequantize(*weighted)[:16] - w[:16]).pow(2).mean()
        assert err_weighted < err_plain


class TestFootprintInvariance:
    def test_weighting_does_not_change_bpw(self):
        # the guarantee the whole matched-footprint comparison rests on
        torch.manual_seed(8)
        w = torch.randn(256, 64)
        sw = torch.rand(256) + 0.1
        codes_p, cb_p = pq_quantize(w, 4, 64)
        codes_w, cb_w = pq_quantize(w, 4, 64, sample_weight=sw)
        assert codes_p.shape == codes_w.shape
        assert cb_p.shape == cb_w.shape
        assert pq_bpw(256, 64, 4, 64) == pq_bpw(256, 64, 4, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_codebook.py::TestWeightedProductQuantization -v`
Expected: FAIL — `TypeError: pq_quantize() got an unexpected keyword argument 'sample_weight'`

- [ ] **Step 3: Implement `pq_quantize`**

Replace the body of `pq_quantize` in `src/smart_quant/codebook.py`. Note the subsample index is hoisted to `sel` so points and weights use the **same** indices — inlining it twice is the silent-corruption bug this guards against:

```python
def pq_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    n_centroids: int,
    iters: int = 10,
    share_codebook: bool = True,
    max_fit: int | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Product-quantize a (out, in) weight. Returns (codes (out, groups) long, codebooks).
    With share_codebook, one codebook is fit over all sub-vectors -> codebooks is
    (n_centroids, sub_dim); otherwise one per group -> (groups, n_centroids, sub_dim).
    max_fit caps how many sub-vectors the shared codebook is *fit* on (a strided subsample);
    all sub-vectors are still assigned. This keeps k-means tractable at scale — fit cost
    drops from ~500k points/expert to max_fit, while assignment is a single pass.

    `sample_weight` (out,) is one importance per input channel; every sub-vector of a row
    inherits its row's weight, since sub-vectors split the *output* dim."""
    out, in_ = weight.shape
    if in_ % sub_dim:
        raise ValueError(f"in_features {in_} not divisible by sub_dim {sub_dim}")
    groups = in_ // sub_dim
    subvecs = weight.reshape(out, groups, sub_dim)

    if share_codebook:
        pool = subvecs.reshape(-1, sub_dim).float()
        pool_w = (None if sample_weight is None
                  else sample_weight.to(pool.device).float().repeat_interleave(groups))
        fit, fit_w = pool, pool_w
        if max_fit is not None and pool.shape[0] > max_fit:
            sel = torch.linspace(0, pool.shape[0] - 1, max_fit).round().long()
            fit = pool[sel]
            fit_w = None if pool_w is None else pool_w[sel]
        centroids = lloyd_kmeans(fit, n_centroids, iters, sample_weight=fit_w)[0]
        idx = torch.cdist(pool, centroids).argmin(dim=1)
        return idx.reshape(out, groups), centroids.to(weight.dtype)

    codes = torch.empty(out, groups, dtype=torch.long)
    codebooks = torch.empty(groups, n_centroids, sub_dim, dtype=weight.dtype)
    for g in range(groups):
        centroids, idx = lloyd_kmeans(
            subvecs[:, g, :].float(), n_centroids, iters, sample_weight=sample_weight)
        codebooks[g] = centroids.to(weight.dtype)
        codes[:, g] = idx
    return codes, codebooks
```

`repeat_interleave(groups)` is correct because `subvecs.reshape(-1, sub_dim)` is row-major: all of row 0's groups, then all of row 1's.

- [ ] **Step 4: Implement `residual_pq_quantize`**

Add the parameter and forward it. Weights apply unchanged at every stage — the residual has the same rows, so the same per-input-channel importance holds:

```python
def residual_pq_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    stage_centroids: list[int],
    iters: int = 10,
    share_codebook: bool = True,
    max_fit: int | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Multi-stage residual product quantization: stage 0 quantizes `weight`, each later
    stage quantizes the running residual `weight - sum(recon so far)`. Returns per-stage
    (codes_list, codebooks_list); `stage_centroids=[k]` reproduces a single `pq_quantize`.
    `sample_weight` applies unchanged at every stage — the residual keeps the same rows."""
    codes_list: list[torch.Tensor] = []
    codebooks_list: list[torch.Tensor] = []
    residual = weight
    last = len(stage_centroids) - 1
    for stage, k in enumerate(stage_centroids):
        codes, codebook = pq_quantize(residual, sub_dim, k, iters, share_codebook, max_fit,
                                      sample_weight=sample_weight)
        codes_list.append(codes)
        codebooks_list.append(codebook)
        if stage < last:  # final stage's residual is never read — skip the dequantize
            residual = residual - pq_dequantize(codes, codebook).to(weight.dtype)
    return codes_list, codebooks_list
```

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 45 passed (39 + 6 new cases).

- [ ] **Step 6: Commit**

```bash
git add src/smart_quant/codebook.py tests/test_codebook.py
git commit -m "feat: sample_weight through pq_quantize and residual_pq_quantize"
```

---

### Task 4: Per-expert weights through the encode path

**Files:**
- Modify: `src/smart_quant/encode.py:37-56` (`quantize_fused_experts`), `src/smart_quant/encode.py:59-91` (`quantize_experts`)
- Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: `residual_pq_quantize(..., sample_weight=...)` from Task 3.
- Produces:
  - `quantize_fused_experts(weight, bits_per_expert, sub_dim, iters=10, codebook_order=1, sample_weight: torch.Tensor | None = None) -> tuple[float, int]` — `sample_weight` is `(d_in,)` (shared by all experts) or `(num_experts, d_in)` (per expert).
  - `quantize_experts(model, avg_bits, sub_dim=4, freqs=None, iters=10, bits_lo=1.5, bits_hi=3.0, codebook_order=1, importance: dict[str, torch.Tensor] | None = None) -> list[dict[str, Any]]` — `importance` is keyed by full parameter name, e.g. `model.layers.0.mlp.experts.gate_up_proj`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_encode.py`:

```python
class TestWeightedFusedExperts:
    def test_footprint_identical_to_unweighted(self):
        # the matched-comparison guarantee: weighting moves centroids, never bit counts
        torch.manual_seed(0)
        base = torch.randn(2, 256, 64)
        plain, weighted = base.clone(), base.clone()
        bits_p, n_p = quantize_fused_experts(plain, torch.full((2,), 2.0), sub_dim=4, iters=5)
        bits_w, n_w = quantize_fused_experts(
            weighted, torch.full((2,), 2.0), sub_dim=4, iters=5,
            sample_weight=torch.rand(2, 256) + 0.1)
        assert (bits_p, n_p) == (bits_w, n_w)
        assert not torch.equal(plain, weighted)   # but the reconstruction differs

    def test_shared_1d_weight_broadcasts_to_every_expert(self):
        torch.manual_seed(1)
        w = torch.randn(3, 128, 32)
        bits, n = quantize_fused_experts(w, torch.full((3,), 2.0), sub_dim=4, iters=5,
                                         sample_weight=torch.rand(128) + 0.1)
        assert n == 3 * 128 * 32
        assert torch.isfinite(w).all()

    def test_heavy_channels_reconstruct_better(self):
        torch.manual_seed(2)
        base = torch.randn(1, 128, 32)
        sw = torch.ones(1, 128)
        sw[0, :16] = 50.0
        plain, weighted = base.clone(), base.clone()
        quantize_fused_experts(plain, torch.full((1,), 1.5), sub_dim=4, iters=15)
        quantize_fused_experts(weighted, torch.full((1,), 1.5), sub_dim=4, iters=15,
                               sample_weight=sw)
        assert ((weighted[0, :16] - base[0, :16]).pow(2).mean()
                < (plain[0, :16] - base[0, :16]).pow(2).mean())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_encode.py::TestWeightedFusedExperts -v`
Expected: FAIL — `TypeError: quantize_fused_experts() got an unexpected keyword argument 'sample_weight'`

- [ ] **Step 3: Implement `quantize_fused_experts`**

```python
def quantize_fused_experts(
    weight: torch.Tensor, bits_per_expert: torch.Tensor, sub_dim: int, iters: int = 10,
    codebook_order: int = 1, sample_weight: torch.Tensor | None = None,
) -> tuple[float, int]:
    """Fake-quantize a fused (num_experts, d_in, d_out) weight in place along the expert dim,
    each expert at its own `bits_per_expert` budget. With `codebook_order > 1` the budget is
    split evenly across that many residual stages. Returns (realized_bits, n_weights): total
    stored bits (indices + shared codebooks, via `pq_bpw`) and the weight count.

    `sample_weight` is per-input-channel importance, either (d_in,) shared across experts or
    (num_experts, d_in) per expert. It changes where centroids land, never the bit count."""
    realized_bits, n_weights = 0.0, 0
    for e in range(weight.shape[0]):
        k = centroids_for_bits(float(bits_per_expert[e]) / codebook_order, sub_dim)
        stage_centroids = [k] * codebook_order  # even split — every stage the same codebook size
        max_fit = max(4096, k * 8)
        sw = None if sample_weight is None else (
            sample_weight if sample_weight.dim() == 1 else sample_weight[e])
        codes, codebooks = residual_pq_quantize(
            weight[e], sub_dim, stage_centroids, iters=iters, max_fit=max_fit, sample_weight=sw)
        weight[e] = pq_dequantize(codes, codebooks).to(weight.dtype)
        out, in_ = weight[e].shape
        realized_bits += pq_bpw(out, in_, sub_dim, stage_centroids) * out * in_
        n_weights += out * in_
    return realized_bits, n_weights
```

- [ ] **Step 4: Implement `quantize_experts`**

The fused-parameter comprehension currently discards names (`encode.py:71`); it must keep them so importance can be looked up per parameter, because `gate_up_proj` and `down_proj` have different `d_in`:

```python
def quantize_experts(
    model, avg_bits: float, sub_dim: int = 4, freqs: dict[str, torch.Tensor] | None = None,
    iters: int = 10, bits_lo: float = 1.5, bits_hi: float = 3.0, codebook_order: int = 1,
    importance: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Walk a model's fused MoE expert modules and fake-quantize them. With `freqs` (router
    name -> usage frequencies), per-expert bits are water-filled to `avg_bits` by usage within
    [bits_lo, bits_hi]; otherwise every expert gets `avg_bits`. `importance` maps a full fused
    parameter name to its per-input-channel weights. Returns per-layer stats."""
    freq_by_layer = {layer_index(k): v for k, v in freqs.items()} if freqs else {}
    stats: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not type(module).__name__.endswith("Experts"):
            continue
        fused = [(pn, p) for pn, p in module.named_parameters(recurse=False) if p.dim() == 3]
        if not fused:
            continue
        n_experts = fused[0][1].shape[0]
        freq = freq_by_layer.get(layer_index(name))
        if freq is not None and len(freq) == n_experts:
            bits = bits_from_frequency(freq, avg_bits, lo=bits_lo, hi=bits_hi)
        else:
            bits = torch.full((n_experts,), float(avg_bits))
        layer_bits, layer_weights = 0.0, 0
        with torch.no_grad():
            for param_name, weight in fused:
                sw = None if importance is None else importance.get(f"{name}.{param_name}")
                fused_bits, fused_weights = quantize_fused_experts(
                    weight, bits, sub_dim, iters, codebook_order=codebook_order, sample_weight=sw)
                layer_bits += fused_bits
                layer_weights += fused_weights
        stats.append({"layer": name, "n_experts": n_experts,
                      "bits_min": round(float(bits.min()), 2),
                      "bits_max": round(float(bits.max()), 2),
                      "quant_bits": layer_bits, "quant_weights": layer_weights})
    return stats
```

Add `from typing import Any` to the imports at the top of `encode.py`.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 48 passed (45 + 3 new cases).

- [ ] **Step 6: Commit**

```bash
git add src/smart_quant/encode.py tests/test_encode.py
git commit -m "feat: per-input-channel importance weights through the encode path"
```

---

### Task 5: Activation importance profiler and shrinkage

Read `docs/plans/2026-07-29-phase7-recon-findings.md` from Task 1 first. If its verdict is **FALLBACK**, implement only the `gate_up_proj` statistic in Step 4 and skip the intermediate recompute.

**Files:**
- Modify: `src/smart_quant/expert_importance.py`
- Test: `tests/test_expert_importance.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `is_moe_router(name: str, module: nn.Module, num_experts: int) -> bool` — extracted from `ExpertUsageProfiler.default_predicate` so both profilers share one source of truth.
  - `shrink_importance(raw: torch.Tensor, counts: torch.Tensor, layer_stat: torch.Tensor, tau: float = 1000.0, alpha: float = 1.0) -> torch.Tensor`
  - `ActivationImportanceProfiler(model, num_experts: int, top_k: int, granularity: str = "expert", router_predicate=None)` with `__enter__`/`__exit__` and `.importance() -> dict[str, torch.Tensor]` keyed by full fused parameter name.

**Do not import `layer_index` from `encode.py`** — `encode.py` already imports `bits_from_frequency` from this module, so that would be a circular import. Router and `Experts` modules are paired by their shared parent name instead: `...layers.N.mlp.gate` and `...layers.N.mlp.experts` both yield `...layers.N.mlp` under `name.rsplit(".", 1)[0]`.

- [ ] **Step 1: Write the failing shrinkage tests**

Add to `tests/test_expert_importance.py`:

```python
class TestShrinkImportance:
    def test_zero_count_expert_falls_back_to_layer_stat(self):
        raw = torch.rand(3, 8) + 0.1
        counts = torch.tensor([0.0, 5000.0, 5000.0])
        layer = torch.rand(8) + 0.1
        w = shrink_importance(raw, counts, layer, tau=1000.0)
        assert torch.allclose(w[0], layer / layer.mean(), atol=1e-6)

    def test_high_count_expert_approaches_raw(self):
        raw = torch.rand(2, 8) + 0.1
        counts = torch.tensor([1e7, 1e7])
        layer = torch.rand(8) + 0.1
        w = shrink_importance(raw, counts, layer, tau=1000.0)
        assert torch.allclose(w[0], raw[0] / raw[0].mean(), atol=1e-3)

    def test_every_row_normalized_to_mean_one(self):
        raw = (torch.rand(4, 16) + 0.1) * 1e5
        w = shrink_importance(raw, torch.full((4,), 100.0), torch.rand(16) + 0.1)
        assert torch.allclose(w.mean(dim=1), torch.ones(4), atol=1e-5)

    def test_alpha_compresses_dynamic_range(self):
        raw = torch.tensor([[1.0, 100.0, 10000.0]])
        counts = torch.tensor([1e7])
        layer = torch.ones(3)
        full = shrink_importance(raw, counts, layer, alpha=1.0)
        soft = shrink_importance(raw, counts, layer, alpha=0.5)
        assert soft.max() / soft.min() < full.max() / full.min()
```

Update the import line to `from smart_quant.expert_importance import (ActivationImportanceProfiler, ExpertUsageProfiler, bits_from_frequency, shrink_importance)`.

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_expert_importance.py::TestShrinkImportance -v`
Expected: FAIL — `ImportError: cannot import name 'shrink_importance'`

- [ ] **Step 3: Implement `shrink_importance`**

Append to `src/smart_quant/expert_importance.py`:

```python
def shrink_importance(
    raw: torch.Tensor,
    counts: torch.Tensor,
    layer_stat: torch.Tensor,
    tau: float = 1000.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Empirical-Bayes shrink per-expert E[x^2] (n_experts, d_in) toward the layer statistic
    (d_in,) by routed-token count, then normalize each expert's vector to mean 1.

    At top-8 of 256 the tail experts see few tokens, so their raw statistic is noise; `tau` is
    the pseudo-count at which raw and layer contribute equally. `alpha < 1` compresses the
    dynamic range, the lever for when unmoderated magnitudes make k-means degenerate."""
    n = counts.to(raw.dtype).unsqueeze(1)
    w = (n * raw + tau * layer_stat.to(raw.dtype).unsqueeze(0)) / (n + tau)
    if alpha != 1.0:
        w = w.clamp(min=0).pow(alpha)
    return w / w.mean(dim=1, keepdim=True).clamp(min=1e-12)
```

- [ ] **Step 4: Extract the shared router predicate**

Add above `ExpertUsageProfiler` in `src/smart_quant/expert_importance.py`, and have the existing method delegate to it so there is one source of truth for the matching rule that survived the transformers-5 refactor:

```python
def is_moe_router(name: str, module: nn.Module, num_experts: int) -> bool:
    """The router is the MoE block's `gate`. transformers >=5 makes it a custom
    Qwen3MoeTopKRouter (not an nn.Linear); <=4 makes it nn.Linear(hidden, num_experts).
    Match by name across both, with an out_features fallback. `...mlp.shared_expert_gate`
    does not end in "mlp.gate", so the shared-expert gate is excluded."""
    if name.endswith("mlp.gate"):
        return True
    return isinstance(module, nn.Linear) and module.out_features == num_experts
```

Then replace `ExpertUsageProfiler.default_predicate`'s body with:

```python
    def default_predicate(self, name: str, module: nn.Module) -> bool:
        return is_moe_router(name, module, self.num_experts)
```

- [ ] **Step 5: Implement `ActivationImportanceProfiler`**

Append to `src/smart_quant/expert_importance.py`:

```python
class ActivationImportanceProfiler:
    """Accumulates E[x^2] of each fused expert projection's input over a calibration set.

    Hooks each `Experts` module for its input hidden states and each router for the per-token
    expert selection, pairing them by shared parent name so the statistic can be attributed
    per expert. Only running sums are retained — never raw activations — so memory is bounded
    at n_layers x n_experts x d_in floats.

    `granularity="layer"` accumulates one vector per projection over all tokens; `"expert"`
    accumulates per routed expert and needs `shrink_importance` before use.
    """

    def __init__(self, model: nn.Module, num_experts: int, top_k: int,
                 granularity: str = "expert", router_predicate=None):
        self.model = model
        self.num_experts = num_experts
        self.top_k = top_k
        self.granularity = granularity
        self.sumsq: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.selection: dict[str, torch.Tensor] = {}
        self.predicate = router_predicate or (
            lambda name, module: is_moe_router(name, module, num_experts))
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def block_of(name: str) -> str:
        """Shared parent of a layer's router and experts: both `...mlp.gate` and
        `...mlp.experts` reduce to `...mlp`, which pairs them without importing
        `layer_index` from `encode` (that direction would be a circular import)."""
        return name.rsplit(".", 1)[0]

    def accumulate(self, key: str, x: torch.Tensor, selection: torch.Tensor | None) -> None:
        """x is (..., d_in); selection is (tokens, top_k) expert ids, or None for layer mode."""
        flat = x.detach().float().reshape(-1, x.shape[-1])
        sq = flat.pow(2).cpu()
        if self.granularity == "layer" or selection is None:
            total = sq.sum(0)
            prev = self.sumsq.get(key)
            self.sumsq[key] = total if prev is None else prev + total
            n = torch.tensor([float(flat.shape[0])])
            self.counts[key] = n if key not in self.counts else self.counts[key] + n
            return
        acc = self.sumsq.get(key)
        cnt = self.counts.get(key)
        if acc is None:
            acc = torch.zeros(self.num_experts, flat.shape[-1])
            cnt = torch.zeros(self.num_experts)
        ones = torch.ones(sq.shape[0])
        for slot in range(selection.shape[1]):
            ids = selection[:, slot].reshape(-1).cpu()
            acc.index_add_(0, ids, sq)
            cnt.index_add_(0, ids, ones)
        self.sumsq[key], self.counts[key] = acc, cnt

    def router_hook(self, name: str):
        def hook(_module, _inp, output):
            logits = output[0] if isinstance(output, tuple) else output
            self.selection[self.block_of(name)] = logits.topk(self.top_k, dim=-1).indices.reshape(
                -1, self.top_k)
        return hook

    def experts_hook(self, name: str):
        def hook(_module, args):
            self.accumulate(f"{name}.gate_up_proj", args[0],
                            self.selection.get(self.block_of(name)))
        return hook

    def __enter__(self) -> "ActivationImportanceProfiler":
        for name, module in self.model.named_modules():
            if self.predicate(name, module):
                self.handles.append(module.register_forward_hook(self.router_hook(name)))
            elif type(module).__name__.endswith("Experts"):
                self.handles.append(
                    module.register_forward_pre_hook(self.experts_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def importance(self) -> dict[str, torch.Tensor]:
        """Mean x^2 per key: (d_in,) in layer mode, (n_experts, d_in) in expert mode."""
        out: dict[str, torch.Tensor] = {}
        for key, total in self.sumsq.items():
            c = self.counts[key].clamp(min=1.0)
            out[key] = total / (c.unsqueeze(-1) if total.dim() == 2 else c)
        return out
```

Hook registration order matters: PyTorch fires hooks in registration order within a module, and `named_modules()` yields the router (`...mlp.gate`) before the experts (`...mlp.experts`) since it walks in definition order — so `self.selection` is populated before `experts_hook` reads it. **Verify this against the recon note's module listing; if `experts` precedes `gate`, the first calibration row silently attributes to a stale/absent selection.** The `test_expert_mode_attributes_to_routed_experts` case in Step 6 catches this on CPU before the box run.

If Task 1's verdict is **FULL**, also recompute the intermediate inside `experts_hook` using the gate/up packing recorded in the findings note and accumulate it under `f"{name}.down_proj"`. If **FALLBACK**, ship `gate_up_proj` only.

- [ ] **Step 6: Write the profiler tests against the existing TinyMoE fixture**

Extend `tests/test_expert_importance.py`, reusing the `model` fixture already in that file:

```python
class TestActivationImportance:
    def test_layer_mode_tracks_input_magnitude(self, model):
        # channel 0 driven 10x harder must dominate the accumulated E[x^2]
        x = torch.randn(4, 6, model.hidden)
        x[..., 0] *= 10.0
        with ActivationImportanceProfiler(model, num_experts=4, top_k=2,
                                          granularity="layer") as prof:
            model(x)
        stat = next(iter(prof.importance().values()))
        assert stat[0] > 5 * stat[1:].mean()

    def test_expert_mode_attributes_to_routed_experts(self, model):
        # guards hook ordering: if the experts hook fires before the router hook,
        # selection is empty and everything silently lands in layer mode
        with ActivationImportanceProfiler(model, num_experts=4, top_k=2) as prof:
            model(torch.randn(4, 6, model.hidden))
        assert prof.selection, "router hook never populated a selection"
        stat = next(iter(prof.importance().values()))
        assert stat.dim() == 2 and stat.shape[0] == 4     # (n_experts, d_in), not collapsed
        counts = next(iter(prof.counts.values()))
        assert counts.sum() == 4 * 6 * 2                  # tokens x top_k attributions

    def test_hooks_removed_on_exit(self, model):
        with ActivationImportanceProfiler(model, num_experts=4, top_k=2) as prof:
            pass
        assert prof.handles == []
```

If `TinyMoE` has no attribute exposing hidden width, add `self.hidden = hidden` in its `__init__` rather than hardcoding the number in the test. If `TinyMoELayer` has no submodule whose class name ends in `Experts`, rename its expert container to `TinyExperts` so the profiler's `endswith("Experts")` match fires — the same rule `quantize_experts` uses.

- [ ] **Step 7: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 55 passed (48 + 7 new cases).

- [ ] **Step 8: Commit**

```bash
git add src/smart_quant/expert_importance.py tests/test_expert_importance.py
git commit -m "feat: activation importance profiler with cold-expert shrinkage"
```

---

### Task 6: CLI wiring

**Files:**
- Modify: `src/smart_quant/cli.py:72-128` (`encode-eval`), and add a `profile-activations` command after `profile-experts`

**Interfaces:**
- Consumes: `ActivationImportanceProfiler`, `shrink_importance`, `quantize_experts(..., importance=...)`.
- Produces: two CLI options on `encode-eval` and one new command. No Python API other code depends on.

- [ ] **Step 1: Add the `profile-activations` command**

Append to `src/smart_quant/cli.py` before `if __name__ == "__main__":`, mirroring `profile-experts`:

```python
@app.command("profile-activations")
def profile_activations(
    model: str = typer.Option(..., help="HF repo id or local path."),
    granularity: str = typer.Option("expert", help="expert | layer."),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    tau: float = typer.Option(1000.0, help="Shrinkage pseudo-count (expert granularity)."),
    alpha: float = typer.Option(1.0, help="Dynamic-range compression, w**alpha."),
    out: Path = typer.Option(Path("experiments/expert_act_importance.pt")),
) -> None:
    """Accumulate per-input-channel E[x^2] for the fused expert projections."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    from smart_quant.expert_importance import ActivationImportanceProfiler, shrink_importance

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModel.from_pretrained(model, torch_dtype="auto", device_map="cuda").eval()
    text_cfg = lm.config.get_text_config()
    rows = load_dataset("allenai/c4", "en", split="train", streaming=True)

    with ActivationImportanceProfiler(
        lm, num_experts=text_cfg.num_experts, top_k=text_cfg.num_experts_per_tok,
        granularity=granularity,
    ) as prof:
        for _, row in zip(range(calib_rows), rows):
            ids = tok(row["text"], return_tensors="pt", truncation=True,
                      max_length=seq_len).input_ids.to("cuda")
            with torch.no_grad():
                lm(ids)
        stats = prof.importance()

    if granularity == "expert":
        stats = {k: shrink_importance(v, prof.counts[k], v.mean(dim=0), tau=tau, alpha=alpha)
                 for k, v in stats.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)
    console.print(f"profiled {len(stats)} expert tensors ({granularity}) "
                  f"over {calib_rows} rows → {out}")
```

- [ ] **Step 2: Add the two `encode-eval` options**

Add these parameters to the `encode_eval` signature, after `allocation`:

```python
    importance_path: Path = typer.Option(None, help="Activation importance .pt from profile-activations."),
    importance_granularity: str = typer.Option("expert", help="expert | layer (row label only)."),
```

Load and pass it, immediately after the existing `freqs = ...` line:

```python
    importance = torch.load(importance_path, weights_only=True) if importance_path else None
    stats = quantize_experts(lm, avg_bits=avg_bits, sub_dim=sub_dim, freqs=freqs,
                             bits_lo=bits_lo, bits_hi=bits_hi, codebook_order=codebook_order,
                             importance=importance)
```

And record it in the row dict so `results.jsonl` is self-describing:

```python
           "importance": importance_granularity if importance_path else None,
```

- [ ] **Step 3: Verify the CLI still loads**

Run: `PYTHONPATH=src .venv/bin/python -c "from smart_quant.cli import app; print('ok')"`
Expected: prints `ok`. (No unit test — the CLI body imports `transformers`, which the CI surface excludes.)

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 55 passed, unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/cli.py
git commit -m "feat: profile-activations command and importance options on encode-eval"
```

---

### Task 7: Plot the weighted curve

**Files:**
- Modify: `experiments/plot_quality_vs_bpw.py`

**Interfaces:**
- Consumes: `results.jsonl` rows whose `label` starts with `wpq`.
- Produces: `weighted_curve(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]`, matching the shape of `uniform_curve` and `residual_curve`.

- [ ] **Step 1: Add the curve selector**

Add next to `residual_curve`. The existing selectors both miss `wpq*-{expert,layer}` labels — `uniform_curve` requires `endswith("-uniform")`, `residual_curve` requires `startswith("rvq")` — so this is purely additive:

```python
def weighted_curve(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """(bpw, ppl, label) for each activation-weighted first-order encode, sorted by footprint.
    Every `wpq*` row carries a realized `expert_bpw`. Empty when no weighted rows exist."""
    pts = [(r["expert_bpw"], r["wikitext_ppl"], r["label"])
           for r in rows if r["label"].startswith("wpq")]
    return sorted(pts)
```

- [ ] **Step 2: Draw it**

Alongside the existing `ax.plot` calls for the first-order and residual curves, add a third in a distinct colour from the blue first-order and purple residual lines:

```python
    weighted = weighted_curve(rows)
    if weighted:
        xs = [p[0] for p in weighted]
        ys = [p[1] for p in weighted]
        ax.plot(xs, ys, "o-", color="tab:green", lw=2, ms=7,
                label="weighted PQ (activation)", zorder=3)
```

- [ ] **Step 3: Verify it regenerates unchanged before any weighted encodes exist**

Run: `PYTHONPATH=src .venv/bin/python experiments/plot_quality_vs_bpw.py`
Expected: same summary line as before — `PQ 6.193 vs imatrix 6.379 @ 2.6 bpw · gap 3.2 pp`. With no `wpq*` rows the new curve is empty and nothing is drawn.

- [ ] **Step 4: Commit**

```bash
git add experiments/plot_quality_vs_bpw.py
git commit -m "feat: weighted-PQ curve on the quality-vs-bpw plot"
```

---

### Task 8: Calibration and encodes on the box

No TDD cycle — this is the experiment. All four runs are sequential because only one fp16 model fits in 80 GB.

**Files:**
- Produces: rows in `experiments/bits-per-brain/results.jsonl` (gitignored), regenerated `experiments/progress/bits-per-brain/quality-vs-bpw.png`

- [ ] **Step 1: Write the sequential launcher**

Create `run_wpq_seq.sh` on the box (not committed — matches how `run_rvq_seq.sh` was handled in Phase 6a):

```bash
#!/usr/bin/env bash
set -euo pipefail
M=Qwen/Qwen3.6-35B-A3B
S="PYTHONPATH=src .venv/bin/python -m smart_quant.cli"

uv run smart-quant profile-activations --model $M --granularity expert \
  --out experiments/expert_act_importance_expert.pt
uv run smart-quant profile-activations --model $M --granularity layer \
  --out experiments/expert_act_importance_layer.pt

for spec in "wpq20-expert 2.0 expert" "wpq25-expert 2.5 expert" "wpq25-layer 2.5 layer"; do
  set -- $spec
  uv run smart-quant encode-eval --model $M --label "$1" --avg-bits "$2" \
    --importance-path "experiments/expert_act_importance_$3.pt" \
    --importance-granularity "$3"
done
```

- [ ] **Step 2: Launch detached, PPID=1**

```bash
setsid nohup bash run_wpq_seq.sh </dev/null \
  >>logs/wpq_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
ps -ef | grep run_wpq_seq | grep -v grep
```

Confirm column 3 (PPID) is `1`. If it is the shell's PID the run will die with the session.

- [ ] **Step 3: Verify the footprint guarantee on the first row**

As soon as `wpq20-expert` appends, check that its realized `expert_bpw` equals `pq2-uniform`'s:

```bash
grep -E '"label": "(pq2-uniform|wpq20-expert)"' experiments/bits-per-brain/results.jsonl \
  | python -c 'import sys,json; [print(json.loads(l)["label"], json.loads(l)["expert_bpw"]) for l in sys.stdin]'
```

Expected: both `2.0`. **A mismatch means the weighting changed the bit accounting — stop and fix before the remaining encodes, because the matched comparison is void.**

- [ ] **Step 4: Regenerate the plot and commit the PNG**

```bash
PYTHONPATH=src .venv/bin/python experiments/plot_quality_vs_bpw.py
git add experiments/progress/bits-per-brain/quality-vs-bpw.png
git commit -m "docs: Phase-7 weighted-PQ quality-vs-bpw plot"
```

`experiments/**/*.png` is blanket-ignored; the negation `!experiments/progress/**/*.png` must already be on its own line in `.gitignore` (gitignore has no inline comments). It was added in Phase 5 — verify with `git check-ignore -v experiments/progress/bits-per-brain/quality-vs-bpw.png` returning nothing.

---

### Task 9: Write up and open the PR

**Files:**
- Modify: `docs/experiments/bits-per-brain.md`

- [ ] **Step 1: Add the Phase-7 section**

After the Phase-6 section, following the same shape (hypothesis, matched-footprint table, verdict, design link). Fill the table from `results.jsonl`:

```markdown
### Phase 7 — activation-weighted first-order PQ

| footprint | uniform PQ | weighted (per-expert) | weighted (per-layer) |
|---|---|---|---|
| ~2.0 bpw | pq2 6.77 | wpq20-expert _ppl_ | — |
| ~2.5 bpw | pq25 6.21 | wpq25-expert _ppl_ | wpq25-layer _ppl_ |
```

State the verdict plainly whichever way it goes. If both weighted arms lose, say so and connect it to Phase 3 and Phase 6a: uniform first-order PQ has now resisted three separate attempts at non-uniform allocation. If the recon verdict was **FALLBACK**, record that `down_proj` was held uniform.

- [ ] **Step 2: Run the suite one final time**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 55 passed.

- [ ] **Step 3: Simplify pass before the PR**

Per the repo convention, run `/simplify` on the working diff and fold findings into the commits they belong to — **before** pushing, not after.

- [ ] **Step 4: Commit, push, open the PR**

```bash
git add docs/experiments/bits-per-brain.md
git commit -m "docs: Phase-7 activation-weighted PQ results"
git push -u origin feat/weighted-pq-phase7
```

Then `gh pr create` with the single-quoted heredoc form, the standard section order (Summary / Test plan / Visual aid / Commits / Out-of-scope follow-ups), grouped commit tables, and every symbol deep-linked to `../tree/feat/weighted-pq-phase7/<path>`. Render-check with `gh pr view <N> --json body --jq '.body' | head -40`.

## Out of scope

- Pre-scaling (per-row `√w`) variant — the natural follow-up if weighting shows signal.
- Phase 6b vptq spike — still deferred.
- AQLM joint beam-search.
- `.pre-commit-config.yaml` ruff C901/PLR0915 gate — repo-wide follow-up.
