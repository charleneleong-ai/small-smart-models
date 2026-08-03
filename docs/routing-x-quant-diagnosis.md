# Routing × quantization — the router survives 2 bpw; allocation wins on routed inputs too

**Study:** bits-per-brain · **Date:** 2026-08-03 · **Design:** [`docs/specs/2026-08-03-routing-x-quant-design.md`](specs/2026-08-03-routing-x-quant-design.md)
**Reproduce:** `PYTHONPATH=src python experiments/diagnose_routing_x_quant.py` (defaults: 512 C4 rows × 2048, layer 13, 32 experts, 2.0 bpw, span [1.5, 3.0])

## What was asked

Item #2 left two questions. **Arm A:** why did the 2×2 reconstruction probe predict usage
allocation would *hurt* shared-codebook PQ (+1.0%) when the corrected end-to-end row won −2.3% ppl?
The probe scored reconstruction on uniformly-sampled weight rows, which cannot weight an expert by
its real routed traffic. **Arm B:** the MoE-unique structural concern — quantization corrupts hidden
states, and the router consumes hidden states, so a quantized model might *re-route*. No part of
the study had measured whether the router changes its selections under quantization, or how deep
the effect runs.

## Arm A — routed-input reconstruction

Run fp16 over the Phase-2 calibration slice, capture the layer-13 hidden rows the router actually
sent each expert (up to 32k tokens each), fold them into a per-expert input Gram `G_e = X_e^T X_e`,
and score a quantized weight by output error `tr((W'_e − W_e) G_e (W'_e − W_e)^T) /
tr(W_e G_e W_e^T)` — the quantity perplexity is sensitive to, not weight-rows-per-unit. Two
allocation cells at matched storage footprint (2.0 bpw, span [1.5, 3.0]), two codebook-size
protocols:

| k protocol | metric | uniform 2.0 | usage-alloc | Δ |
|---|---|---|---|---|
| pow2 (shipped `centroids_for_bits`) | odd-row relL2 sum (2×2 protocol) | 10.5226 | 10.2141 | **−2.93%** |
| pow2 | routed out err (token-weighted) | 0.2573 | 0.2194 | **−14.71%** |
| int `2^(bits·4)` (the 2×2 probe's) | odd-row relL2 sum | 10.5226 | 10.6230 | **+0.95%** |
| int | routed out err | 0.2573 | 0.2273 | **−11.63%** |

Three things fall out:

1. **The routed measure settles item #2's open question.** Weighted by what the router actually
   computes, allocation helps — under *both* k protocols (−11.6% / −14.7%), matching the sign of
   the corrected end-to-end row (−2.3% ppl). The magnitude is larger than ppl because routed
   reconstruction error is not ppl (the two are correlated, not equal).
2. **The 2×2 probe's wrong sign was a k-mapping artifact, not the routing blind spot.** The int
   row reproduces the 2×2's published numbers exactly (10.5226 / 10.6230 / +0.95% vs its +1.0%).
   Swap in the shipped `centroids_for_bits` — `2^round(bits·4)` instead of `int(2^(bits·4))` — and
   the identical proxy flips to −2.93%. Fractional water-filled bits land on different codebook
   sizes between the two mappings (e.g. 2.125 → 256 vs 362 centroids), and the uniform-row proxy is
   sensitive to that; the routed measure is not.
3. **The blind spot was real, just not sign-determining.** The routed measure does more than
   confirm the direction the corrected proxy already shows — it says allocation's value concentrates
   on the experts that run, which the odd-row protocol can only understate.

## Arm B — router drift under quantization

Fake-quantized the whole model at 2 bpw uniform (realized 2.010 expert bpw), replayed the same 512
rows, compared per-layer top-8 selections against fp16, with a *content-change* null (fp16 on
disjoint row halves — what routing would change anyway):

| layer | slot agree | top-1 agree | freq L1 | null L1 (content) | Jaccard | null Jaccard |
|---|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.000 | 0.053 | 1.000 | 1.000 |
| 12 | 0.378 | 0.775 | 0.064 | 0.167 | 1.000 | 0.778 |
| 24 | 0.344 | 0.741 | 0.070 | 0.157 | 0.600 | 0.600 |
| 38 | 0.438 | 0.783 | 0.062 | 0.128 | 0.600 | 0.600 |
| range 1–39 | 0.34–0.58 | 0.74–0.88 | 0.036–0.070 | 0.05–0.22 | 0.33–1.0 | 0.33–1.0 |

- **The router is robust.** Layer 0 is untouched (agreement 1.000 — nothing above it was
  quantized). Below that, top-1 selections agree with fp16 74–88% of the time and per-slot
  agreement sits far above the 8/256 = 3% random baseline, declining through the mid layers
  (worst ≈ layer 24–25) before recovering in the deepest layers.
- **Quantization moves routing *less* than the text does.** Selection-frequency L1 under
  quantization (0.04–0.07) is smaller than the content-change null (0.05–0.22) at every layer. The
  hot expert sets overlap comparably. A quantized model's routing is closer to the fp16 model's
  routing on the *same* tokens than fp16 routing is across two slices of text.
- **No compounding, no self-healing.** Layer-13 per-expert correlation between routed output error
  and fp16→quant frequency change is r = −0.086 — no evidence the router systematically flees
  badly-quantized experts or piles onto them.

## Verdict

- **The MoE-unique structural concern does not materialize at 2 bpw.** A quantized router keeps
  picking the same experts; the calibration-time routing distribution the allocation is built on
  stays valid. Quantizing an MoE behaves like quantizing a dense net plus per-expert error — the
  "keep the router path clean" mitigation from the design is not warranted by this data.
- **Item #2 is closed on the right measure.** Usage allocation helps shared-codebook PQ at a fixed
  byte budget whether you score reconstruction on routed inputs (−12 to −15%) or read it in ppl
  (−2.3%). The 2×2 probe's contrary +1.0% was its codebook-size mapping, reproduced exactly here.
- **The uniform-row proxy is protocol-fragile.** Its sign depends on how fractional water-filled
  bits map to k; the routed measure is stable across both mappings. Any follow-up reconstruction
  proxy should score routed inputs and use the shipped k mapping.

## Caveats

- Arm A is layer-13 `gate_up_proj`, first-32 experts, first-of-256 slice — the 2×2 convention.
  Arm B covers the full 40 layers.
- One operating point only: 2 bpw uniform. The drift-vs-null margin is wide at every layer, but a
  more aggressive rate (or the 1.5–3.0 allocation) could move routing further; worth one run if
  the study pushes below 2 bpw.
- The routed measure is reconstruction-weighted, not ppl; its 14.7% versus the ppl −2.3% gap is
  the usual proxy transfer.
- The content null compares routing across *text*; it is not a same-magnitude perturbation. It is
  the right "would change anyway" baseline, and quantization clears it comfortably.

## Next move

- **One robustness run at the aggressive end** (avg 1.5 bpw or the 1.5–3.0 allocation) if the
  study targets lower rates — confirm the drift-vs-null margin holds before relying on router
  stability as a general property.
- The scalar-ppl transfer question stays deferred (item #2: no scalar end-to-end build).
