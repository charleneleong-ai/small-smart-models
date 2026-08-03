# Allocation 2×2 — why usage allocation helps GGUF but (mostly) not our PQ

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
aggressive span realizes a smaller tensor, and a smaller tensor scores worse. The end-to-end
verdict ("usage allocation hurts") survives, but as a comparison at *lower* footprint, not at
matched footprint. The correct matched comparison is the table above: allocation costs PQ **+1.0%**
reconstruction error, not +13.6% perplexity.

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

## Verdict

- **The mechanism claim survives, weakly.** Fit-limited PQ under-responds to reallocation; a
  rate-limited scalar family over-responds. Directionally confirmed on real tensors.
- **The study's headline negative was over-stated.** Phase 3's magnitude was substantially footprint
  drift; the true matched-footprint cost of usage allocation for PQ is ~1% reconstruction error at
  layer 13. That is still a loss, and uniform PQ remains the local optimum — but the reason
  "the field is right and we are right" is not that MoE allocation is a myth. It is that the field
  holds its byte budget fixed and the study's water-fill did not.

## Caveats

Reconstruction error on layer-13 `gate_up_proj`, 32 experts, first-of-256 slice. Proxy discipline,
as throughout this study — Phase 7 is the standing reminder that a proxy sign can fail to transfer
to perplexity. Here the proxy argues *for* a correction (Phase 3 was confounded), so the risk is an
under-correction, not an over-claim.

## Next move

Re-run `pq2-expert` **end-to-end with a true arithmetic-footprint allocation** (target 2.0 bpw) to
put a clean perplexity number on the corrected claim. `quantize_experts` already accepts a
`bits_per_expert` tensor; the fix is applying the same arithmetic re-centring in the encode path.
A scalar end-to-end build is only justified if the corrected PQ row still leaves the interaction
worth chasing as a perplexity result — Phase 7's record says assume it will not transfer.
