# Phase 9 — E8 Lattice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fake-quantize expert weights to the E8 lattice at a target bpw and measure whether ~10% better reconstruction becomes better perplexity.

**Architecture:** Round each 8-wide sub-vector to its nearest E8 point at a per-tensor scale, bisected to hit a target rate. Codes are never stored (fake quantization), so no shell enumeration or encoder is needed — only honest rate accounting: `bpw = ceil(log2(distinct points))/8`, with zero codebook term.

**Tech Stack:** Python 3.13, PyTorch, typer, pytest.

**Spec:** [`docs/specs/2026-08-01-e8-lattice-phase9-design.md`](../specs/2026-08-01-e8-lattice-phase9-design.md)

## Global Constraints

- Branch `feat/e8-lattice-phase9`, off `main`. One PR. Conventional commits, no `Co-Authored-By`.
- Type hints with explicit generics on every new signature; no leading underscores on module-level names.
- CI surface is torch-only — no test may import `transformers` or `datasets`.
- Absent `lattice=True`, every existing path stays **byte-identical**.
- `PYTHONPATH=src .venv/bin/python -m pytest -q` before each commit. Baseline **57 passed** on `main`.

---

### Task 1: E8 nearest-point and distinct counting

**Files:** Create `src/smart_quant/lattice.py`, `tests/test_lattice.py`

**Produces:**
- `nearest_e8(x: torch.Tensor) -> torch.Tensor` — `(n, 8)` → nearest E8 point
- `distinct_points(pts: torch.Tensor) -> int` — integer-hash count

- [ ] **Step 1: failing tests**

```python
import pytest
import torch

from smart_quant.lattice import distinct_points, nearest_e8


def on_e8(p: torch.Tensor) -> torch.Tensor:
    """E8 = D8 union (D8 + 1/2): all-integer or all-half-odd coordinates, even sum either way."""
    is_int = torch.allclose(p, p.round())
    is_half = torch.allclose(p - 0.5, (p - 0.5).round())
    return (is_int or is_half) and bool(((p.sum(1) % 2).abs() < 1e-6).all())


class TestNearestE8:
    def test_returns_lattice_points(self):
        torch.manual_seed(0)
        assert on_e8(nearest_e8(torch.randn(512, 8) * 3))

    def test_lattice_points_are_fixed(self):
        # a point already on the lattice must map to itself
        torch.manual_seed(1)
        pts = nearest_e8(torch.randn(256, 8) * 3)
        assert torch.allclose(nearest_e8(pts), pts)

    def test_matches_brute_force_on_a_neighbourhood(self):
        # enumerate D8 and D8+1/2 within a small box and check we pick the true nearest
        torch.manual_seed(2)
        grid = torch.cartesian_prod(*[torch.arange(-1.0, 2.0)] * 4)
        pad = torch.zeros(grid.shape[0], 4)
        d8 = torch.cat([grid, pad], 1)
        d8 = d8[d8.sum(1) % 2 == 0]
        book = torch.cat([d8, d8 + 0.5])
        x = torch.rand(64, 8) - 0.5                       # inside the enumerated region
        brute = book[torch.cdist(x, book).argmin(1)]
        ours = nearest_e8(x)
        assert torch.allclose((ours - x).pow(2).sum(1), (brute - x).pow(2).sum(1), atol=1e-5)


class TestDistinctPoints:
    def test_counts_known_distinct_points(self):
        pts = nearest_e8(torch.randn(4096, 8, generator=torch.Generator().manual_seed(3)) * 2)
        uniq = torch.unique(pts, dim=0)
        assert distinct_points(pts) == uniq.shape[0]

    def test_repeats_do_not_inflate_the_count(self):
        pts = nearest_e8(torch.randn(128, 8, generator=torch.Generator().manual_seed(4)) * 2)
        assert distinct_points(pts.repeat(5, 1)) == distinct_points(pts)
```

- [ ] **Step 2:** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_lattice.py -q` → FAIL, no module.

- [ ] **Step 3: implement** `src/smart_quant/lattice.py`

```python
"""E8 lattice quantization.

A stored codebook costs O(2^{kd}*d) to keep and to search, which is why our shape sweep found
k-means at sub_dim=8 costing 6.0 bpw with 4.0 of it codebook. A lattice escapes that: its points
are computed, not tabulated, so sub_dim=8 is reachable at zero storage.

E8 is the densest sphere packing in 8 dimensions and is what QuIP# uses. Nearest-point search is
Conway & Sloane's: E8 = D8 union (D8 + 1/2), and the nearest D8 point is coordinate-wise rounding
with a parity fix. O(d) per sub-vector, no search.

This module supports *fake* quantization only — the reconstruction is written back and codes are
never stored, so no canonical shell enumeration or encoder is needed. Rate is accounted from the
measured number of distinct points used.
"""
from __future__ import annotations

import torch

__all__ = ["nearest_e8", "distinct_points"]

HASH_BASE = 40507  # prime, comfortably above any doubled-coordinate magnitude we produce


def nearest_d8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of D8 (integer coordinates, even sum): round, then if the sum came out odd,
    re-round the single coordinate whose rounding was least confident."""
    r = torch.round(x)
    odd = (r.sum(1) % 2 != 0)
    if odd.any():
        err = x - r
        j = err.abs().argmax(dim=1)
        rows = torch.arange(x.shape[0], device=x.device)
        flip = torch.sign(err[rows, j])
        flip[flip == 0] = 1.0
        r[rows[odd], j[odd]] += flip[odd]
    return r


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Nearest E8 point to each row of x (n, 8). E8 is the union of D8 and its half-shift, so
    take whichever coset point is closer."""
    a = nearest_d8(x)
    b = nearest_d8(x - 0.5) + 0.5
    closer = (x - a).pow(2).sum(1) <= (x - b).pow(2).sum(1)
    return torch.where(closer.unsqueeze(1), a, b)


def distinct_points(pts: torch.Tensor) -> int:
    """Number of distinct lattice points, via an integer polynomial hash. Coordinates are
    half-integers, so doubling makes them integral; `unique(dim=0)` on tens of millions of rows is
    not viable, and a 1-D unique over hashes is."""
    key = (pts * 2).to(torch.int64)
    powers = torch.tensor([HASH_BASE ** i for i in range(pts.shape[1])],
                          dtype=torch.int64, device=pts.device)
    return int(torch.unique((key * powers).sum(1)).numel())
```

- [ ] **Step 4:** full suite → expect **63 passed** (57 + 6).
- [ ] **Step 5:** commit `feat: E8 nearest-point quantizer and distinct-point counting`

---

### Task 2: Scale calibration and fused quantization

**Files:** Modify `src/smart_quant/lattice.py`, `tests/test_lattice.py`

**Consumes:** `nearest_e8`, `distinct_points`.
**Produces:**
- `calibrate_scale(pool: torch.Tensor, target_bpw: float, sub_dim: int = 8, iters: int = 24) -> float`
- `quantize_e8_fused(weight: torch.Tensor, target_bpw: float, sub_dim: int = 8) -> tuple[float, int]`
  — fake-quantizes in place, returns `(realized_bits, n_weights)`, matching `quantize_fused_experts`.

- [ ] **Step 1: failing tests**

```python
from smart_quant.lattice import calibrate_scale, quantize_e8_fused


class TestCalibrateScale:
    def test_hits_the_target_rate(self):
        torch.manual_seed(5)
        pool = torch.randn(200_000, 8) * 0.01
        s = calibrate_scale(pool, target_bpw=2.5)
        rate = math.ceil(math.log2(distinct_points(nearest_e8(pool / s)))) / 8
        assert abs(rate - 2.5) <= 0.125          # one bit of index granularity

    def test_coarser_scale_gives_fewer_points(self):
        # monotonicity is what makes bisection valid
        torch.manual_seed(6)
        pool = torch.randn(50_000, 8) * 0.01
        counts = [distinct_points(nearest_e8(pool / s)) for s in (0.004, 0.008, 0.016)]
        assert counts[0] > counts[1] > counts[2]


class TestQuantizeE8Fused:
    def test_writes_reconstruction_and_reports_zero_codebook(self):
        torch.manual_seed(7)
        w = torch.randn(2, 64, 128) * 0.01
        orig = w.clone()
        bits, n = quantize_e8_fused(w, target_bpw=2.5)
        assert n == 2 * 64 * 128
        assert not torch.equal(w, orig) and torch.isfinite(w).all()
        # realized rate must be the measured index cost, with no codebook term
        assert 2.0 <= bits / n <= 3.0

    def test_higher_target_gives_lower_error(self):
        torch.manual_seed(8)
        base = torch.randn(1, 64, 128) * 0.01
        lo, hi = base.clone(), base.clone()
        quantize_e8_fused(lo, target_bpw=2.0)
        quantize_e8_fused(hi, target_bpw=3.0)
        assert (hi - base).pow(2).mean() < (lo - base).pow(2).mean()

    def test_requires_sub_dim_divisibility(self):
        with pytest.raises(ValueError):
            quantize_e8_fused(torch.randn(1, 8, 20), target_bpw=2.5)
```

- [ ] **Step 2:** run → FAIL, names not defined.

- [ ] **Step 3: implement** (append to `lattice.py`, add `math` import and both names to `__all__`)

```python
def calibrate_scale(pool: torch.Tensor, target_bpw: float, sub_dim: int = 8,
                    iters: int = 24, max_fit: int = 2_000_000) -> float:
    """Scale whose distinct-point count realizes `target_bpw`.

    Distinct count decreases monotonically in the scale, so bisection converges. Calibrated on a
    strided subsample — the caller re-measures the realized rate on the full tensor, so a
    subsample that lands slightly off-target is visible rather than silent."""
    fit = pool if pool.shape[0] <= max_fit else pool[
        torch.linspace(0, pool.shape[0] - 1, max_fit).round().long()]
    target_points = 2.0 ** (target_bpw * sub_dim)
    lo, hi = 1e-5, 1.0
    for _ in range(iters):
        mid = (lo * hi) ** 0.5                       # geometric: scale spans orders of magnitude
        if distinct_points(nearest_e8(fit / mid)) > target_points:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def quantize_e8_fused(weight: torch.Tensor, target_bpw: float,
                      sub_dim: int = 8) -> tuple[float, int]:
    """Fake-quantize a fused (num_experts, out, in) weight to the E8 lattice in place.

    One scale for the whole tensor: that is what the pooled measurement used, and it beat 32
    per-expert k-means codebooks. Returns (realized_bits, n_weights) with **no codebook term** —
    lattice points are computed, so the only cost is the index and one fp16 scale."""
    if sub_dim != 8:
        raise ValueError("E8 is defined for sub_dim=8")
    n_experts, out, in_ = weight.shape
    if in_ % sub_dim:
        raise ValueError(f"in_features {in_} not divisible by sub_dim {sub_dim}")

    pool = weight.reshape(-1, sub_dim).float()
    scale = calibrate_scale(pool, target_bpw, sub_dim)
    pts = nearest_e8(pool / scale)
    weight.copy_((pts * scale).reshape(weight.shape).to(weight.dtype))

    index_bits = math.ceil(math.log2(max(distinct_points(pts), 2)))
    n_weights = weight.numel()
    return index_bits * (n_weights / sub_dim) + 16, n_weights
```

- [ ] **Step 4:** full suite → expect **68 passed**.
- [ ] **Step 5:** commit `feat: per-tensor scale calibration and fused E8 quantization`

---

### Task 3: Encode-path and CLI wiring

**Files:** Modify `src/smart_quant/encode.py`, `src/smart_quant/cli.py`, `tests/test_encode.py`

- [ ] **Step 1: failing test** in `tests/test_encode.py`

```python
class TestLatticeEncode:
    def test_lattice_flag_changes_the_result_and_default_is_untouched(self):
        torch.manual_seed(0)
        base = TinyExperts(num_experts=2, hidden=16, inter=8)
        plain = wrap_in_layers(copy.deepcopy(base), prefix_depth=1)
        latt = wrap_in_layers(copy.deepcopy(base), prefix_depth=1)
        quantize_experts(plain, avg_bits=2.5, iters=3)
        quantize_experts(latt, avg_bits=2.5, iters=3, lattice=True)
        pe = plain.inner.layers[0].mlp.experts
        le = latt.inner.layers[0].mlp.experts
        assert not torch.equal(pe.gate_up_proj, le.gate_up_proj)
        assert torch.isfinite(le.gate_up_proj).all()
```

`TinyExperts(hidden=16, inter=8)` gives `gate_up_proj` in_=16 and `down_proj` in_=8 — both divisible by 8.

- [ ] **Step 2:** run → FAIL, unexpected kwarg `lattice`.

- [ ] **Step 3: implement.** In `quantize_experts`, add `lattice: bool = False` and dispatch per fused parameter:

```python
                if lattice:
                    fused_bits, fused_weights = quantize_e8_fused(weight, avg_bits)
                else:
                    fused_bits, fused_weights = quantize_fused_experts(
                        weight, bits, sub_dim, iters, codebook_order=codebook_order,
                        channel_weight=cw)
```

Import `quantize_e8_fused` from `smart_quant.lattice`. Document in the docstring that `lattice=True`
uses `avg_bits` as a *target* rate realized by scale calibration, and ignores `freqs`/`importance`
since a lattice takes no calibration statistic.

- [ ] **Step 4: CLI.** Add `lattice: bool = typer.Option(False, help="Quantize to the E8 lattice instead of a learned codebook.")` to `encode_eval`, pass it through, and record `"quantizer": "e8" if lattice else "pq"` on the results row.

- [ ] **Step 5:** full suite → expect **69 passed**. Verify CLI loads.
- [ ] **Step 6:** commit `feat: E8 lattice option on the encode path and CLI`

---

### Task 4: Encodes on the box

- [ ] **Step 1:** push; `ssh pi-a100-80gb` fetch, checkout, `pytest -q` → 69 passed.
- [ ] **Step 2:** smoke-test the CLI path end-to-end with a tiny run before the full encode.
- [ ] **Step 3:** runner `run_e8_seq.sh` — **`rc=$?` on its own line**, never `exit=$?` inside an echo with a `$(...)`, which silently masked three OOMs in Phase 8:

```bash
export PYTHONPATH=src
PY=.venv/bin/python
M=Qwen/Qwen3.6-35B-A3B
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run() {
  echo "[seq] $(ts) START $1"
  $PY -m smart_quant.cli encode-eval --model "$M" --label "$1" --avg-bits "$2" --lattice
  rc=$?
  echo "[seq] $(ts) DONE $1 exit=$rc"
}
run e8-25 2.5
run e8-20 2.0
echo "[seq] $(ts) ALL-DONE"
```

- [ ] **Step 4:** launch detached (`setsid nohup ... </dev/null >>logs/e8_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &`), confirm SID == PID.
- [ ] **Step 5: footprint gate.** As soon as `e8-25` appends, check its realized `expert_bpw` is **≤ 2.542**. If calibration drifted above the k-means baseline, the comparison is void — stop and re-calibrate before `e8-20` runs.

---

### Task 5: Write up and PR

- [ ] **Step 1:** Phase-9 section in `docs/experiments/bits-per-brain.md` — hypothesis, matched-footprint table, verdict. If perplexity does not follow the ~10% reconstruction gain, say so plainly: that is the fourth time the proxy failed to transfer and belongs in the [local-optimum note](../local-optimum.md) as a headline, not a caveat.
- [ ] **Step 2:** add the `e8` family to `FAMILIES` in `experiments/plot_quality_vs_bpw.py`; regenerate; commit the PNG.
- [ ] **Step 3:** `/simplify` on the working diff, fold findings into their commits **before** pushing.
- [ ] **Step 4:** push, `gh pr create` with the standard section order and deep links, then render-check.

## Out of scope

Canonical shell enumeration and a real encoder (deployment, not measurement); incoherence processing on the lattice (Phase 10 — bundling would make a win unattributable); per-expert scales; trellis/QTIP quantizers.
