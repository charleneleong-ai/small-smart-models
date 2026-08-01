# Phase 9 — E8 lattice quantization

**Study:** bits-per-brain · **Branch:** `feat/e8-lattice-phase9` (off `main`)
**Date:** 2026-08-01 · **Status:** design pending review

## Problem

[Nine measurements](../local-optimum.md) say uniform shared-codebook PQ at `sub_dim=4` is a strong
local optimum: four allocation phases lost end-to-end, and utilization, fitting quality, column
rescaling, incoherence processing and codebook shape all have nothing left to give. The tenth
measurement is the only one that escapes — an **E8 lattice** gives ~8–11% better reconstruction MSE
at matched 2.53 bpw, from a single *global* lattice with no per-expert adaptation, beating 32
individually fitted codebooks.

That gain has never been run end-to-end. Phase 9 tests whether ~10% reconstruction MSE becomes
perplexity — the transfer this study has repeatedly watched fail.

## Why a lattice escapes the wall

A stored codebook costs `O(2^{kd}·d)` to keep *and* search. Our own shape sweep shows it directly:
k-means at `sub_dim=8, k=65536` costs **6.0 bpw, 4.0 of which is codebook storage**, and still loses
to `sub_dim=2` at 3.0 bpw. E8 — the densest sphere packing in 8 dimensions, and what QuIP# uses —
makes `sub_dim=8` reachable at **zero** storage, because its points are computed rather than
tabulated.

Nearest-point search is the classical Conway & Sloane algorithm: `E8 = D8 ∪ (D8 + ½)`, where the
nearest D8 point is coordinate-wise rounding with a parity fix. O(d) per sub-vector, no search.

## The simplification that makes this small

This study does **fake quantization** — it writes the dequantized reconstruction back into the fp16
weights and measures perplexity. **Codes are never stored.** So Phase 9 needs no canonical shell
enumeration, no index assignment, and no encoder: those are deployment concerns, not measurement
concerns.

What it does need is honest rate accounting. The realized rate is

```
bpw = ceil(log2(distinct lattice points used)) / sub_dim        # + one fp16 scale per tensor
```

with **zero** codebook term. Distinct points are counted on the full tensor, so the reported bpw is
measured rather than assumed.

## Design

### `src/smart_quant/lattice.py` (new)

- `nearest_e8(x: torch.Tensor) -> torch.Tensor` — rounds `(n, 8)` to the nearest E8 point.
- `distinct_points(pts: torch.Tensor) -> int` — count via an integer polynomial hash of `2·pts`
  (coordinates are half-integers, so doubling makes them integral); a 67M-row `unique(dim=0)` is
  not viable.
- `calibrate_scale(pool, target_bpw, ...) -> float` — bisection on `s`. Distinct count is monotone
  decreasing in `s`, so bisection converges. Calibrated on a strided subsample for speed; the
  *reported* rate is then recomputed on the full tensor.
- `quantize_e8_fused(weight, target_bpw, ...) -> tuple[float, int]` — mirrors
  `quantize_fused_experts`' contract: fake-quantizes in place, returns `(realized_bits, n_weights)`.

### Scale granularity: one per fused tensor

One `s` shared by all 256 experts in a tensor. That is what the pooled measurement used, and it
already beat 32 per-expert k-means codebooks. Storage is one fp16 per tensor — free. Per-expert
scales would need 256 bisections per tensor and are deferred.

### Both projections, unlike Phase 8

`gate_up_proj` has `in_=2048` and `down_proj` has `in_=512`; both are divisible by 8, so E8 applies
to each. Phase 8 had to restrict to `gate_up_proj` because its per-layer Hessian only existed for
the hidden-state input; a lattice needs no calibration statistic at all, so that restriction lifts.

### Wiring

- [`encode.py`](../../src/smart_quant/encode.py) — `quantize_experts(..., lattice: bool = False)`.
  When set, each fused tensor goes through `quantize_e8_fused` instead of the PQ path. Absent it,
  every existing path is byte-identical.
- [`cli.py`](../../src/smart_quant/cli.py) — `--lattice` flag on `encode-eval`. Labels `e8-*`.

## Experiment

| label | target bpw | pairs against |
|---|---|---|
| `e8-25` | 2.5 | `pq25-uniform` (2.542) **6.2137** |
| `e8-20` | 2.0 | `pq2-uniform` (~2.010) **6.765** |

Sequential on `pi-a100-80gb`. No calibration pass is needed — E8 uses no activation statistics —
which makes this the cheapest phase in the study.

**Success:** `e8-25 < 6.2137` at realized bpw `<= 2.542`. That would be the first technique in the
study to beat uniform PQ, and it would do so with *less* machinery, not more.

**Failure is still informative:** if ~10% reconstruction MSE does not become perplexity, that is the
fourth time this study has caught the reconstruction proxy failing to transfer, and it would make
that a headline finding in its own right rather than a caveat.

## Risks

- **Transfer is the main risk.** Phase 7 improved its own objective by 41% and *lost* perplexity.
  Reconstruction MSE is a proxy, and this study's record with it is poor.
- **Calibration drift.** A scale bisected on a subsample may land off-target on the full tensor. The
  encode reports *measured* distinct counts, so an off-target rate is visible rather than silent;
  the footprint gate below catches it before the second encode runs.
- **Per-tensor scale may be too coarse.** Experts differ; one scale for 256 of them may lose more
  than the lattice gains. The pooled measurement suggests not, but it covered 32 experts of one
  layer.

## Testing

New cases in `tests/test_lattice.py`:

- **E8 membership** — every returned point has even coordinate-sum in `2·x` terms, i.e. lies in
  `D8 ∪ (D8 + ½)`.
- **Nearest-point correctness** — against brute force over a small enumerated E8 neighbourhood on
  random inputs.
- **Rounding is idempotent** — a point already on the lattice maps to itself.
- **Distinct-count hash has no collisions** on a constructed set of known-distinct points.
- **`calibrate_scale` hits its target** within a stated tolerance, and is monotone in `s`.
- **Footprint accounting** — `quantize_e8_fused` reports zero codebook storage, and its realized bpw
  matches `ceil(log2(distinct))/sub_dim` to within the scale term.
- **Absent `lattice=True`, `quantize_experts` is byte-identical** to today.

## Out of scope

- **Canonical shell enumeration and a real encoder** — needed for deployment, not for fake-quantized
  measurement.
- **Incoherence processing on top of the lattice.** The [local-optimum note](../local-optimum.md)
  predicts rotation *should* pay off here where it did not for k-means, since a fixed codebook cannot
  adapt to the data's shape. That is the natural Phase 10 and is deliberately not bundled, so this
  phase tests one thing.
- **Per-expert scales**, trellis/QTIP-style quantizers, and `down_proj`-specific tuning.
