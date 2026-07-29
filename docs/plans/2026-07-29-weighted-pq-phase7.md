# Phase 7 — Activation-Weighted First-Order PQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add activation-derived importance weighting to the first-order product quantizer and measure whether it beats uniform PQ at matched footprint.

**Architecture:** Weighting is applied by change of variables — scale each input channel by `√w`, fit the codebook in the scaled space, divide back out on reconstruction. This exactly minimizes the weighted objective while leaving `lloyd_kmeans`, `pq_quantize` and `pq_dequantize` untouched; all of it lives in `quantize_fused_experts`. The one footprint cost is storing `w`, counted honestly by a new `scale_len` term on `pq_bpw`.

**Tech Stack:** Python 3.13, PyTorch (CPU for unit tests, CUDA on the box), typer CLI, pytest, matplotlib.

**Spec:** [`docs/specs/2026-07-29-weighted-pq-phase7-design.md`](../specs/2026-07-29-weighted-pq-phase7-design.md)
**Recon:** [`docs/plans/2026-07-29-phase7-recon-findings.md`](2026-07-29-phase7-recon-findings.md) — Task 1, complete, verdict **FULL**

## Global Constraints

- Branch `feat/weighted-pq-phase7`, off `main`. Never commit to `main`; the phase lands as one PR.
- Conventional commits (`feat:`, `test:`, `docs:`). No `Co-Authored-By` trailers.
- Type hints on every new signature, params **and** return, with explicit generics: `dict[str, torch.Tensor]`, never bare `dict`.
- No leading underscores on module-level functions, classes, or constants.
- Tests go in the **existing** files by area — `tests/test_codebook.py`, `tests/test_encode.py`, `tests/test_expert_importance.py`. Do not create new test modules.
- The CI surface is torch-only (`pyproject.toml` `[dependency-groups] test`). No test may import `transformers` or `datasets`.
- Every default must preserve existing behaviour: absent weights take the current code path and Phase-5/6 encodes stay byte-identical.
- Run the full suite with `PYTHONPATH=src .venv/bin/python -m pytest -q` before each commit. Baseline is **35 passed**.

### Confirmed facts from recon — do not re-derive

- Fused weights are `(num_experts, out_features, in_features)`: `gate_up_proj (256, 1024, 2048)`, `down_proj (256, 2048, 512)`. Sub-vectors run along the **input** dim.
- `Experts.forward(hidden_states, top_k_index, top_k_weights)` — routing arrives as an argument, so a single `forward_pre_hook` suffices. No router hook, no layer pairing.
- `hidden_states` is `(tokens, hidden)` already flattened; `top_k_index` is `(tokens, top_k)`.
- Intermediate recompute: `gate, up = F.linear(x_e, gate_up_proj[e]).chunk(2, -1)`, then `act_fn(gate) * up`.
- Config: `num_experts=256`, `top_k=8`, `hidden=2048`, `moe_intermediate_size=512`.

---

### ~~Task 1: Box reconnaissance~~ — COMPLETE

Committed as `55db620`. Verdict **FULL**: both projections get real statistics.

---

### Task 2: `pq_bpw` scale-vector term

**Files:**
- Modify: `src/smart_quant/codebook.py:115-127`
- Test: `tests/test_codebook.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `pq_bpw(out: int, in_: int, sub_dim: int, n_centroids: int | list[int], share_codebook: bool = True, scale_len: int | None = None) -> float`. `scale_len` is the number of fp16 scale values stored alongside the codes; `None` reproduces today's result exactly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_codebook.py`, inside the existing `TestBpw` class:

```python
    def test_scale_len_none_matches_today(self):
        assert pq_bpw(1024, 2048, 4, 256, scale_len=None) == pq_bpw(1024, 2048, 4, 256)

    @pytest.mark.parametrize("out,in_,expected", [(1024, 2048, 0.015625), (2048, 512, 0.0078125)])
    def test_scale_term_matches_hand_computed_overhead(self, out, in_, expected):
        # w is one fp16 per input channel: in_ * 16 bits over out * in_ weights
        plain = pq_bpw(out, in_, 4, 256)
        scaled = pq_bpw(out, in_, 4, 256, scale_len=in_)
        assert scaled - plain == pytest.approx(expected, rel=1e-9)
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_codebook.py::TestBpw -v`
Expected: FAIL — `TypeError: pq_bpw() got an unexpected keyword argument 'scale_len'`

- [ ] **Step 3: Implement**

```python
def pq_bpw(
    out: int, in_: int, sub_dim: int, n_centroids: int | list[int],
    share_codebook: bool = True, scale_len: int | None = None,
) -> float:
    """Effective bits-per-weight including fp16 codebook storage, summed over residual stages.
    `n_centroids` may be a single int (one stage) or a per-stage list. Sharing a single codebook
    (vs one per group) is what drops the overhead from ~index-storage to negligible.

    `scale_len` counts fp16 per-input-channel scales stored alongside the codes, as the
    activation-weighted encode needs to undo its change of variables at reconstruction."""
    stages = n_centroids if isinstance(n_centroids, list) else [n_centroids]
    groups = in_ // sub_dim
    n_codebooks = 1 if share_codebook else groups
    total_bits = sum(
        out * groups * math.log2(k) + n_codebooks * k * sub_dim * 16 for k in stages
    )
    if scale_len is not None:
        total_bits += scale_len * 16
    return total_bits / (out * in_)
```

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 38 passed (35 baseline + 3 new cases).

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/codebook.py tests/test_codebook.py
git commit -m "feat: scale_len term on pq_bpw for weighted-encode storage"
```

---

### Task 3: Scaled-space quantization in the encode path

The core of the phase.

**Files:**
- Modify: `src/smart_quant/encode.py` — module docstring (lines 1-11, the `(d_in, d_out)` claim is wrong), `quantize_fused_experts` (37-56), `quantize_experts` (59-91)
- Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: `pq_bpw(..., scale_len=...)` from Task 2.
- Produces:
  - `quantize_fused_experts(weight, bits_per_expert, sub_dim, iters=10, codebook_order=1, sample_weight: torch.Tensor | None = None) -> tuple[float, int]` — `sample_weight` is `(in_features,)` shared across experts or `(num_experts, in_features)` per expert.
  - `quantize_experts(model, avg_bits, sub_dim=4, freqs=None, iters=10, bits_lo=1.5, bits_hi=3.0, codebook_order=1, importance: dict[str, torch.Tensor] | None = None) -> list[dict[str, Any]]` — `importance` keyed by full fused parameter name, e.g. `language_model.layers.0.mlp.experts.gate_up_proj`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_encode.py`:

```python
class TestWeightedFusedExperts:
    def test_none_weight_is_byte_identical(self):
        torch.manual_seed(0)
        base = torch.randn(2, 64, 128)
        a, b = base.clone(), base.clone()
        r1 = quantize_fused_experts(a, torch.full((2,), 2.0), sub_dim=4, iters=5)
        r2 = quantize_fused_experts(b, torch.full((2,), 2.0), sub_dim=4, iters=5,
                                    sample_weight=None)
        assert r1 == r2
        assert torch.equal(a, b)

    def test_heavy_input_channels_reconstruct_better(self):
        # channels 0:16 weighted 50x must end up closer than under the unweighted fit,
        # and the ignored channels correspondingly worse — the whole point of the feature
        torch.manual_seed(1)
        base = torch.randn(1, 64, 128)
        sw = torch.ones(128)
        sw[:16] = 50.0
        plain, weighted = base.clone(), base.clone()
        quantize_fused_experts(plain, torch.full((1,), 1.5), sub_dim=4, iters=15)
        quantize_fused_experts(weighted, torch.full((1,), 1.5), sub_dim=4, iters=15,
                               sample_weight=sw)
        err_p = (plain[0, :, :16] - base[0, :, :16]).pow(2).mean()
        err_w = (weighted[0, :, :16] - base[0, :, :16]).pow(2).mean()
        assert err_w < err_p
        rest_p = (plain[0, :, 16:] - base[0, :, 16:]).pow(2).mean()
        rest_w = (weighted[0, :, 16:] - base[0, :, 16:]).pow(2).mean()
        assert rest_w > rest_p

    def test_minimizes_the_weighted_objective(self):
        # the change-of-variables claim: scaled-space fitting must lower total *weighted*
        # error, even though it raises unweighted error
        torch.manual_seed(2)
        base = torch.randn(1, 64, 128)
        sw = torch.rand(128) * 10 + 0.1
        plain, weighted = base.clone(), base.clone()
        quantize_fused_experts(plain, torch.full((1,), 1.5), sub_dim=4, iters=15)
        quantize_fused_experts(weighted, torch.full((1,), 1.5), sub_dim=4, iters=15,
                               sample_weight=sw)
        werr_p = (sw * (plain[0] - base[0]).pow(2)).mean()
        werr_w = (sw * (weighted[0] - base[0]).pow(2)).mean()
        assert werr_w < werr_p

    def test_footprint_includes_the_scale_vector(self):
        torch.manual_seed(3)
        w = torch.randn(2, 64, 128)
        bits_p, n = quantize_fused_experts(w.clone(), torch.full((2,), 2.0), sub_dim=4, iters=5)
        bits_w, n2 = quantize_fused_experts(w.clone(), torch.full((2,), 2.0), sub_dim=4, iters=5,
                                            sample_weight=torch.rand(128) + 0.1)
        assert n == n2
        # one fp16 scale per input channel per expert: 2 experts * 128 * 16 bits
        assert bits_w - bits_p == pytest.approx(2 * 128 * 16, rel=1e-9)

    def test_per_expert_2d_weight_accepted(self):
        torch.manual_seed(4)
        w = torch.randn(3, 64, 128)
        bits, n = quantize_fused_experts(w, torch.full((3,), 2.0), sub_dim=4, iters=5,
                                         sample_weight=torch.rand(3, 128) + 0.1)
        assert n == 3 * 64 * 128
        assert torch.isfinite(w).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_encode.py::TestWeightedFusedExperts -v`
Expected: FAIL — `TypeError: quantize_fused_experts() got an unexpected keyword argument 'sample_weight'`

- [ ] **Step 3: Fix the wrong module docstring**

Replace lines 3-4 of `src/smart_quant/encode.py`. The recon confirmed the layout is the standard `nn.Linear` one, because the forward uses `F.linear(x, W)`:

```python
transformers >=5 stores experts as fused `(num_experts, out_features, in_features)` tensors —
the standard `nn.Linear` layout, since the forward applies `F.linear(x, W)`. Each expert is a 2D
slice `W[e]`. We product-quantize each slice with a shared codebook at a per-expert bit budget —
uniform, or driven by routing usage via `bits_from_frequency` — then write the dequantized
reconstruction back. This "fake quantization" lets the fp16 model reflect quantized weights for
quality measurement without custom inference kernels.
```

- [ ] **Step 4: Implement `quantize_fused_experts`**

Sub-vectors span `sub_dim` consecutive *input* channels, so `sample_weight` broadcasts across columns. The scaled space is computed in float32 for numerical headroom — fp16 `√w` can underflow on cold channels:

```python
def quantize_fused_experts(
    weight: torch.Tensor, bits_per_expert: torch.Tensor, sub_dim: int, iters: int = 10,
    codebook_order: int = 1, sample_weight: torch.Tensor | None = None,
) -> tuple[float, int]:
    """Fake-quantize a fused (num_experts, out_features, in_features) weight in place along the
    expert dim, each expert at its own `bits_per_expert` budget. With `codebook_order > 1` the
    budget is split evenly across that many residual stages. Returns (realized_bits, n_weights).

    `sample_weight` is per-input-channel importance, (in_,) shared across experts or
    (num_experts, in_) per expert. It is applied by change of variables: columns are scaled by
    sqrt(w), the codebook is fit in that space, and the reconstruction is divided back out — which
    exactly minimizes sum_ij w_j (W_ij - West_ij)^2. The scale vector is stored, so `pq_bpw` counts
    it via `scale_len`."""
    realized_bits, n_weights = 0.0, 0
    for e in range(weight.shape[0]):
        k = centroids_for_bits(float(bits_per_expert[e]) / codebook_order, sub_dim)
        stage_centroids = [k] * codebook_order  # even split — every stage the same codebook size
        max_fit = max(4096, k * 8)
        out, in_ = weight[e].shape

        if sample_weight is None:
            target, w_sqrt, scale_len = weight[e], None, None
        else:
            w = sample_weight if sample_weight.dim() == 1 else sample_weight[e]
            w_sqrt = w.to(device=weight.device, dtype=torch.float32).clamp(min=1e-12).sqrt()
            target = weight[e].float() * w_sqrt
            scale_len = in_

        codes, codebooks = residual_pq_quantize(
            target, sub_dim, stage_centroids, iters=iters, max_fit=max_fit)
        recon = pq_dequantize(codes, codebooks)
        weight[e] = (recon if w_sqrt is None else recon / w_sqrt).to(weight.dtype)

        realized_bits += pq_bpw(out, in_, sub_dim, stage_centroids,
                                scale_len=scale_len) * out * in_
        n_weights += out * in_
    return realized_bits, n_weights
```

- [ ] **Step 5: Implement `quantize_experts`**

The fused-parameter comprehension currently discards names (`encode.py:71`); it must keep them, because `gate_up_proj` and `down_proj` have different `in_features` and so need different weight vectors:

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

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 43 passed (38 + 5 new cases).

- [ ] **Step 7: Commit**

```bash
git add src/smart_quant/encode.py tests/test_encode.py
git commit -m "feat: activation-weighted quantization via scaled-space codebook fit"
```

---

### Task 4: Activation importance profiler and shrinkage

**Files:**
- Modify: `src/smart_quant/expert_importance.py`
- Test: `tests/test_expert_importance.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `shrink_importance(raw: torch.Tensor, counts: torch.Tensor, layer_stat: torch.Tensor, tau: float = 1000.0, alpha: float = 1.0) -> torch.Tensor`
  - `ActivationImportanceProfiler(model, num_experts: int, granularity: str = "expert")` with `__enter__`/`__exit__`, `.importance() -> dict[str, torch.Tensor]` keyed by full fused parameter name, and `.counts: dict[str, torch.Tensor]`.

**Do not import `layer_index` from `encode.py`** — `encode.py` already imports `bits_from_frequency` from this module, so that direction would be a circular import. Nothing here needs it.

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

Update the import line to:

```python
from smart_quant.expert_importance import (
    ActivationImportanceProfiler, ExpertUsageProfiler, bits_from_frequency, shrink_importance)
```

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
    the pseudo-count at which raw and layer contribute equally. Normalizing to mean 1 also keeps
    sqrt(w) near unity, so the scaled-space fit stays in a sane numeric range. `alpha < 1`
    compresses the dynamic range, the lever for when unmoderated magnitudes make k-means
    degenerate."""
    n = counts.to(raw.dtype).unsqueeze(1)
    w = (n * raw + tau * layer_stat.to(raw.dtype).unsqueeze(0)) / (n + tau)
    if alpha != 1.0:
        w = w.clamp(min=0).pow(alpha)
    return w / w.mean(dim=1, keepdim=True).clamp(min=1e-12)
```

- [ ] **Step 4: Implement `ActivationImportanceProfiler`**

Append to `src/smart_quant/expert_importance.py`. Accumulation is always per-expert; the layer statistic is that marginalized over experts, so the two arms of the ablation differ *only* in granularity:

```python
class ActivationImportanceProfiler:
    """Accumulates E[x^2] of each fused expert projection's input over a calibration set.

    A single forward-pre hook on each `Experts` module suffices: transformers 5 passes the
    routing in as `forward(hidden_states, top_k_index, top_k_weights)`, so there is no router
    hook to pair with. `gate_up_proj`'s statistic is over the incoming hidden states;
    `down_proj`'s is over the intermediate, recomputed per expert exactly as the forward does.

    Only running sums are retained — never raw activations — so memory is bounded at
    n_layers x n_experts x d_in floats, held on CPU.
    """

    def __init__(self, model: nn.Module, num_experts: int, granularity: str = "expert"):
        self.model = model
        self.num_experts = num_experts
        self.granularity = granularity
        self.sumsq: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(self, key: str, expert: int, x: torch.Tensor) -> None:
        """Fold x (n_tokens, d_in) into the running sum for one expert."""
        sq = x.detach().float().pow(2).sum(0).cpu()
        if key not in self.sumsq:
            self.sumsq[key] = torch.zeros(self.num_experts, sq.shape[0])
            self.counts[key] = torch.zeros(self.num_experts)
        self.sumsq[key][expert] += sq
        self.counts[key][expert] += x.shape[0]

    def make_hook(self, name: str):
        def hook(module, args):
            hidden, top_k_index = args[0], args[1]
            flat = hidden.reshape(-1, hidden.shape[-1])
            for e in torch.unique(top_k_index).tolist():
                if e >= self.num_experts:
                    continue
                token_idx = (top_k_index == e).any(dim=-1).nonzero(as_tuple=True)[0]
                if token_idx.numel() == 0:
                    continue
                x_e = flat[token_idx]
                self.add(f"{name}.gate_up_proj", e, x_e)
                gate, up = nn.functional.linear(x_e, module.gate_up_proj[e]).chunk(2, dim=-1)
                self.add(f"{name}.down_proj", e, module.act_fn(gate) * up)
        return hook

    def __enter__(self) -> "ActivationImportanceProfiler":
        for name, module in self.model.named_modules():
            if type(module).__name__.endswith("Experts"):
                self.handles.append(module.register_forward_pre_hook(self.make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def importance(self) -> dict[str, torch.Tensor]:
        """Mean x^2 per key: (n_experts, d_in) in expert mode, (d_in,) marginalized in layer mode."""
        out: dict[str, torch.Tensor] = {}
        for key, total in self.sumsq.items():
            cnt = self.counts[key]
            if self.granularity == "layer":
                out[key] = total.sum(0) / cnt.sum().clamp(min=1.0)
            else:
                out[key] = total / cnt.clamp(min=1.0).unsqueeze(1)
        return out
```

- [ ] **Step 5: Write the profiler tests**

The existing `TinyMoE` fixture predates this and does not match the transformers-5 `Experts` shape. Add a small fixture alongside it that does — fused params, and a forward taking routing — so the profiler is exercised against the real contract:

```python
class TinyExperts(nn.Module):
    """Mirrors the transformers-5 fused Experts contract the profiler hooks."""

    def __init__(self, num_experts: int = 4, hidden: int = 8, inter: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.1)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return torch.zeros_like(hidden_states)


class TestActivationImportance:
    def test_attributes_per_routed_expert(self):
        torch.manual_seed(0)
        experts = TinyExperts()
        x = torch.randn(10, 8)
        idx = torch.tensor([[0, 1]] * 6 + [[2, 3]] * 4)
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            experts(x, idx, torch.ones(10, 2))
        counts = prof.counts["gate_up_proj"]
        assert counts.tolist() == [6.0, 6.0, 4.0, 4.0]
        assert prof.importance()["gate_up_proj"].shape == (4, 8)

    def test_down_proj_statistic_matches_hand_computed_intermediate(self):
        torch.manual_seed(1)
        experts = TinyExperts()
        x = torch.randn(5, 8)
        idx = torch.zeros(5, 1, dtype=torch.long)      # every token to expert 0
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            experts(x, idx, torch.ones(5, 1))
        gate, up = nn.functional.linear(x, experts.gate_up_proj[0]).chunk(2, dim=-1)
        expected = (experts.act_fn(gate) * up).pow(2).mean(0)
        assert torch.allclose(prof.importance()["down_proj"][0], expected, atol=1e-5)

    def test_layer_mode_marginalizes_over_experts(self):
        torch.manual_seed(2)
        experts = TinyExperts()
        x = torch.randn(6, 8)
        idx = torch.tensor([[0, 1]] * 6)
        with ActivationImportanceProfiler(experts, num_experts=4,
                                          granularity="layer") as prof:
            experts(x, idx, torch.ones(6, 2))
        stat = prof.importance()["gate_up_proj"]
        assert stat.shape == (8,)
        assert torch.allclose(stat, x.pow(2).mean(0), atol=1e-5)

    def test_hooks_removed_on_exit(self):
        experts = TinyExperts()
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            pass
        assert prof.handles == []
```

Note the keys are bare `gate_up_proj` / `down_proj` here because `named_modules()` yields `""` for the root module, so `f"{name}.{param}"` has an empty prefix. On the real model they are fully qualified.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 51 passed (43 + 8 new cases).

- [ ] **Step 7: Commit**

```bash
git add src/smart_quant/expert_importance.py tests/test_expert_importance.py
git commit -m "feat: activation importance profiler with cold-expert shrinkage"
```

---

### Task 5: CLI wiring

**Files:**
- Modify: `src/smart_quant/cli.py` — `encode-eval` (72-128), plus a `profile-activations` command

- [ ] **Step 1: Add the `profile-activations` command**

Append before `if __name__ == "__main__":`, mirroring `profile-experts`:

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
        lm, num_experts=text_cfg.num_experts, granularity=granularity
    ) as prof:
        for _, row in zip(range(calib_rows), rows):
            ids = tok(row["text"], return_tensors="pt", truncation=True,
                      max_length=seq_len).input_ids.to("cuda")
            with torch.no_grad():
                lm(ids)
        stats = prof.importance()
        counts = dict(prof.counts)

    if granularity == "expert":
        stats = {k: shrink_importance(v, counts[k], v.mean(dim=0), tau=tau, alpha=alpha)
                 for k, v in stats.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)
    console.print(f"profiled {len(stats)} expert tensors ({granularity}) "
                  f"over {calib_rows} rows → {out}")
```

- [ ] **Step 2: Add the two `encode-eval` options**

Add to the `encode_eval` signature, after `allocation`:

```python
    importance_path: Path = typer.Option(None, help="Activation importance .pt from profile-activations."),
    importance_granularity: str = typer.Option("expert", help="expert | layer (recorded on the row)."),
```

Load and pass it, replacing the existing `stats = quantize_experts(...)` call:

```python
    importance = torch.load(importance_path, weights_only=True) if importance_path else None
    stats = quantize_experts(lm, avg_bits=avg_bits, sub_dim=sub_dim, freqs=freqs,
                             bits_lo=bits_lo, bits_hi=bits_hi, codebook_order=codebook_order,
                             importance=importance)
```

And record it in the row dict so `results.jsonl` is self-describing — add after `"codebook_order": codebook_order,`:

```python
           "importance": importance_granularity if importance_path else None,
```

- [ ] **Step 3: Verify the CLI still loads**

Run: `PYTHONPATH=src .venv/bin/python -c "from smart_quant.cli import app; print('ok')"`
Expected: prints `ok`. No unit test — the command bodies import `transformers`, which the CI surface excludes.

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 51 passed, unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/cli.py
git commit -m "feat: profile-activations command and importance options on encode-eval"
```

---

### Task 6: Plot the weighted curve

**Files:**
- Modify: `experiments/plot_quality_vs_bpw.py`

- [ ] **Step 1: Add the curve selector**

Add next to `residual_curve`. The existing selectors both miss `wpq*-{expert,layer}` labels — `uniform_curve` requires `endswith("-uniform")`, `residual_curve` requires `startswith("rvq")` — so this is purely additive:

```python
def weighted_curve(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """(bpw, ppl, label) for each activation-weighted first-order encode, sorted by footprint.
    Every `wpq*` row carries a realized `expert_bpw` that includes its stored scale vector.
    Empty when no weighted rows exist."""
    pts = [(r["expert_bpw"], r["wikitext_ppl"], r["label"])
           for r in rows if r["label"].startswith("wpq")]
    return sorted(pts)
```

- [ ] **Step 2: Draw it**

Alongside the existing `ax.plot` calls, in a colour distinct from the blue first-order and purple residual lines:

```python
    weighted = weighted_curve(rows)
    if weighted:
        ax.plot([p[0] for p in weighted], [p[1] for p in weighted], "o-", color="tab:green",
                lw=2, ms=7, label="weighted PQ (activation)", zorder=3)
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

### Task 7: Calibration and encodes on the box

No TDD cycle — this is the experiment. Sequential because only one fp16 model fits in 80 GB.

- [ ] **Step 1: Sync the branch to the box**

The box is at `830919f` (pre-rebase Phase 6a). Fetch and check out this branch there:

```bash
ssh pi-a100-80gb 'cd ~/small-smart-models && git fetch origin && git checkout feat/weighted-pq-phase7 && git log --oneline -1'
```

This requires the branch to be pushed first — push before running Task 7.

- [ ] **Step 2: Write the sequential launcher**

Create `run_wpq_seq.sh` on the box (not committed — matches how `run_rvq_seq.sh` was handled in Phase 6a):

```bash
#!/usr/bin/env bash
set -euo pipefail
M=Qwen/Qwen3.6-35B-A3B

for g in expert layer; do
  uv run smart-quant profile-activations --model $M --granularity $g \
    --out "experiments/expert_act_importance_$g.pt"
done

for spec in "wpq20-expert 2.0 expert" "wpq25-expert 2.5 expert" "wpq25-layer 2.5 layer"; do
  set -- $spec
  uv run smart-quant encode-eval --model $M --label "$1" --avg-bits "$2" \
    --importance-path "experiments/expert_act_importance_$3.pt" \
    --importance-granularity "$3"
done
```

- [ ] **Step 3: Launch detached, PPID=1**

```bash
setsid nohup bash run_wpq_seq.sh </dev/null \
  >>logs/wpq_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
ps -ef | grep run_wpq_seq | grep -v grep
```

Confirm column 3 (PPID) is `1`. If it is the shell's PID the run dies with the session.

- [ ] **Step 4: Verify the footprint delta is the scale vector and nothing else**

As soon as `wpq20-expert` appends:

```bash
grep -E '"label": "(pq2-uniform|wpq20-expert)"' experiments/bits-per-brain/results.jsonl \
  | python -c 'import sys,json; [print(json.loads(l)["label"], json.loads(l)["expert_bpw"]) for l in sys.stdin]'
```

Expected: `pq2-uniform 2.0` and `wpq20-expert` at **~2.012** — the weighted average of +0.0156 (gate_up) and +0.0078 (down) over the two tensors. Materially more than that means something other than the scale vector changed the accounting; stop and diagnose before the remaining encodes.

- [ ] **Step 5: Regenerate the plot and commit the PNG**

```bash
PYTHONPATH=src .venv/bin/python experiments/plot_quality_vs_bpw.py
git add experiments/progress/bits-per-brain/quality-vs-bpw.png
git commit -m "docs: Phase-7 weighted-PQ quality-vs-bpw plot"
```

Verify the PNG is not ignored first: `git check-ignore -v experiments/progress/bits-per-brain/quality-vs-bpw.png` should return nothing.

---

### Task 8: Write up and open the PR

**Files:**
- Modify: `docs/experiments/bits-per-brain.md`

- [ ] **Step 1: Add the Phase-7 section**

After the Phase-6 section, same shape (hypothesis, matched-footprint table, verdict, design link). Fill from `results.jsonl`:

```markdown
### Phase 7 — activation-weighted first-order PQ

| footprint | uniform PQ | weighted (per-expert) | weighted (per-layer) |
|---|---|---|---|
| ~2.0 bpw | pq2 (2.00) 6.77 | wpq20-expert (2.01) _ppl_ | — |
| ~2.5 bpw | pq25 (2.54) 6.21 | wpq25-expert (2.55) _ppl_ | wpq25-layer (2.50) _ppl_ |
```

State the verdict plainly whichever way it goes, and note that the weighted arms carry ~0.012 bpw of stored scale vector — reported, not tuned away. If both weighted arms lose, connect it to Phase 3 and Phase 6a: uniform first-order PQ has now resisted three separate attempts at non-uniform allocation.

- [ ] **Step 2: Run the suite one final time**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 51 passed.

- [ ] **Step 3: Simplify pass before the PR**

Run `/simplify` on the working diff and fold findings into the commits they belong to — **before** pushing, not after.

- [ ] **Step 4: Commit, push, open the PR**

```bash
git add docs/experiments/bits-per-brain.md
git commit -m "docs: Phase-7 activation-weighted PQ results"
git push -u origin feat/weighted-pq-phase7
```

Then `gh pr create` with the single-quoted heredoc form, standard section order (Summary / Test plan / Visual aid / Commits / Out-of-scope follow-ups), grouped commit tables, and every symbol deep-linked to `../tree/feat/weighted-pq-phase7/<path>`. Render-check with `gh pr view <N> --json body --jq '.body' | head -40`.

## Out of scope

- Per-sub-vector scalar weighting — an approximation of the same objective.
- Phase 6b vptq spike — still deferred.
- AQLM joint beam-search.
- `.pre-commit-config.yaml` ruff C901/PLR0915 gate — repo-wide follow-up.
