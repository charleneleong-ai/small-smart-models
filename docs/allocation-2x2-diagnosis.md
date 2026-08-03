# Allocation 2×2 — usage allocation helps GGUF and our PQ, once the byte budget holds

**Study:** bits-per-brain · **Date:** 2026-08-03 · **Design:** [`docs/specs/2026-08-03-allocation-2x2-design.md`](specs/2026-08-03-allocation-2x2-design.md)
**Reproduce:** `PYTHONPATH=src python experiments/diagnose_allocation_2x2.py` (defaults: layer 13, 32 experts, 2.0 bpw, span [1.5, 3.0])

## What was asked

The 2026 landscape allocates more bits to hot MoE experts and reports gains (BitsMoE, MC-MoE,
imatrix). Phase 3 measured the same water-fill *hurting* our shared-codebook PQ. This is the
mechanism question behind that divergence: is allocation's payoff tied to the quantizer's error
structure — **rate-limited** (scalar: closed-form scales, error tracks the rate) vs **fit-limited**
(PQ: Lloyd on a finite sample, error saturates below the promised rate)?

The 2×2 on real layer-13 `gate_up_proj` weights (first 32 experts, disjoint odd/even fit-eval
halves, the real Phase-2 routing frequency, sum of `rel L2`):

| family | uniform | usage-alloc | Δ |
|---|---|---|---|
| scalar per-row | 22.6260 | 21.8205 | **−3.6%** |
| PQ shared-codebook d=4 | 10.5226 | 10.6230 | **+1.0%** |

The interaction is present and in the predicted direction — the same water-fill helps the
rate-limited family and hurts the fit-limited one — but it is small. The bigger finding fell out of
building the matched control.

## The footprint confound — Phase 3's magnitude was an accounting artifact

`bits_from_frequency` pins the **usage-weighted** mean to `avg_bits`. Storage cost is the
**arithmetic** mean. With realistic routing skew the two diverge sharply. Measured on the full
256-expert layer-13 router frequency:

| span | usage-weighted | arithmetic (realized) | drift |
|---|---|---|---|
| [1.5, 3.0] | 2.000 | **1.633** | −18% |
| [1.8, 2.3] | 2.000 | **1.857** | −7% |

So Phase 3's headline — `pq2-uniform` 6.77 ppl < gentle 7.07 < aggressive 7.68 — compared a true
2.0 bpw tensor against ~1.86 and ~1.63 bpw tensors. The **monotone-in-strength** pattern that made
the negative look like a real allocation effect is exactly what footprint drift produces: a more
aggressive span realizes a smaller tensor, and a smaller tensor scores worse. The **monotone
pattern did not survive matched footprint** — the corrected end-to-end row below turns the sign.

This also explains most of the field divergence. BitsMoE and imatrix allocate under a **fixed byte
budget** — levels are assigned so the file lands on size. The study's water-fill did not: it shrank
the tensor and called the difference an allocation loss.

## The mechanism, measured

At a genuinely matched footprint (shift-and-clamp the water-fill to arithmetic mean 2.0), both
spans agree:

| span | scalar Δ | pq Δ |
|---|---|---|
| [1.5, 3.0] | −3.6% | +1.0% |
| [1.8, 2.3] | −2.5% | +0.1% |

Opposite signs, monotone in span strength, deterministic. The R-D curve at uniform rate backs the
mechanism: scalar is steeper at low rates (−8.4 dB/bit at 2.0) while PQ holds a flat ≈ −5.5 dB/bit
across 2.0–3.0; scalar's edge narrows as the rate rises and the two converge by 3.0 bpw. PQ really
is less rate-responsive — but on real weights the gap is a few dB/bit, not the wall the synthetic
probe suggested, which is why the interaction shows as single-digit percent rather than a dramatic
flip.

## The corrected end-to-end row

Re-ran `pq2-expert` with the arithmetic re-centring in the encode path
([`src/smart_quant/expert_importance.py`](../../tree/main/src/smart_quant/expert_importance.py))
so the *storage* mean lands on 2.0 bpw. Same model, same eval, same wikitext-2 protocol:

| row | allocation | target bpw | realized bpw | span | wikitext ppl | Δ vs uniform |
|---|---|---|---|---|---|---|
| `pq2-uniform` | uniform | 2.0 | ~2.0 | [2.0, 2.0] | 6.765 | — |
| `pq2-expert-storage` | usage | 2.0 | **2.099** | [1.8, 3.0] | **6.607** | **−2.3%** |
| `pq2-expert` (Phase 3, confounded) | usage | 2.0 | ~1.63 | [1.5, 3.0] | 7.683 | +13.6% |
| `pq2-expert-gentle` (Phase 3, confounded) | usage | 2.0 | ~1.86 | [1.8, 2.3] | 7.073 | +4.6% |

At matched footprint the allocation **wins** — −2.3% ppl (6.607 vs 6.765) — and it does so while
realizing slightly *more* storage (2.099 vs ~2.0 bpw, the price of per-expert codebooks), so the
true matched-footprint number is at worst that. The confounded rows' magnitude was entirely
footprint; their sign was wrong too.

The 2×2 reconstruction probe predicted the opposite for PQ (+1.0%). That probe sampled rows
uniformly (even/odd halves) to guarantee disjointness, so it could not see what ppl rewards: the
routed computation spends its error budget on the experts that actually run. This is the
Phase-7-shaped lesson in reverse — the reconstruction proxy's sign failed to transfer, this time
by *understating* allocation's value.

## Verdict

- **The mechanism claim holds directionally, but the perplexity answer favours allocation.** The
  2×2 reconstruction table showed the predicted interaction — the same water-fill helps a
  rate-limited scalar family (−3.6%) and slightly hurts a fit-limited PQ (up to +1.0%). The
  corrected end-to-end row overturns the PQ sign at the only scale that matters: matched-footprint
  usage allocation scores **6.607 vs 6.765** (uniform 2.0 bpw). PQ's fit-limitedness shows up as a
  *smaller* payoff than scalar's, not as a loss.
- **Phase 3's headline negative was an accounting artifact, end to end.** The usage-weighted
  water-fill shrank the tensor (~1.63 bpw at a 2.0 target) and the 7.68/7.07 ppl it produced were
  smaller-tensor scores, not allocation losses. The field is right — BitsMoE, MC-MoE and imatrix
  hold their byte budget fixed, and so does the corrected encode; at a fixed budget the allocation
  is a free win. Uniform PQ is *not* the local optimum the study's #1 recorded it as.

## Caveats

Reconstruction numbers are layer-13 `gate_up_proj`, 32 experts, first-of-256 slice. The corrected
row is a full-model encode with the honest `expert_bpw=2.099` recorded — slightly above the 2.0
uniform, so the win is conservative. Proxy discipline throughout: the 2×2 probe predicted a PQ
loss and perplexity found a small win; the uniform-row-sampling proxy cannot see routing-weighted
value. No scalar end-to-end was built, so the scalar arm's transfer is untested.

## Next move

The corrected row closes item #2: allocation works, at a fixed byte budget, for both scalar (proxy)
and PQ (perplexity). Remaining questions, in order:

1. **Item #3 — routing × quantization.** The MoE-unique loss part of the claim (shared codebook
   fit on cold experts). The 2×2's PQ arm already hints the routed-weighted measure is the
   binding one — a routed-input reconstruction probe would resolve the proxy's blind spot.
   **Resolved 2026-08-03** — the routed measure helps allocation (−12 to −15% reconstruction),
   the 2×2's +1.0% was its codebook-size mapping, and the router barely re-routes under 2 bpw
   quantization: [`docs/routing-x-quant-diagnosis.md`](routing-x-quant-diagnosis.md).
2. A scalar end-to-end build is not justified: the probe's scalar win (−3.6% recon) is the same
   ordering ppl shows for PQ, and Phase 7's record says scalar-in-ppl transfers poorly.
