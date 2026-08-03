# Research item #3 — routing × quantization

**Study:** bits-per-brain · **Branch:** `feat/routing-x-quant` (off `main`)
**Date:** 2026-08-03 · **Status:** design pending review

## Problem

Item #2 closed with a sign-flip the study does not yet understand. The 2×2 reconstruction probe
([`allocation-2x2-diagnosis.md`](../allocation-2x2-diagnosis.md)) predicted usage allocation would
*hurt* shared-codebook PQ (+1.0% reconstruction); the corrected end-to-end row showed a **−2.3% ppl
win**. The discrepancy was attributed to the probe's eval protocol: it scores reconstruction on
uniformly-sampled output rows, which cannot weight an expert by how much *routed* traffic actually
touches it. That is item #3's first arm.

The second arm is the MoE-unique structure the whole study's claim rests on. Quantization corrupts
hidden states; the router consumes hidden states to pick experts. A dense model has no such
feedback — its quantization error stays local. An MoE can *re-route* under quantization, which
would (a) make the calibration-time routing distribution stale for allocation, and (b) compound
errors in a way no per-expert footprint argument captures. Nothing in this study has measured
whether the router actually changes its selections when the experts are quantized.

## Arm A — routed-input reconstruction (why the probe was blind)

Replace the uniform-row eval half with the **real routed inputs**: run the fp16 model over the
Phase-2 calibration slice and capture, per expert, the actual `x_e` the router sent it. Score the
quantized weight's *output* error on those inputs — the quantity perplexity is sensitive to.

- Layer-13 `gate_up_proj`, first-32 experts (the diagnose convention), routed tokens captured by a
  pre-hook on the layer-13 `Experts` module (`args[0]` hidden, `args[1]` top-k indices), capped at
  32k routed tokens per expert.
- Codebook fit on **all** weight rows (the eval is on inputs, so there is no in-sample risk), at
  per-expert `k = 2^(4·b_e)` for uniform 2.0 and for the arithmetic-centred usage allocation.
- Metric per expert: output `rel L2` on routed tokens, `‖(W'_e − W_e) x_e‖ / ‖W_e x_e‖`, aggregated
  as the routed-token-weighted mean across experts.
- **Decision:** if allocation now *wins* on the routed measure (uniform-row probe said +1%, ppl
  said −2.3%), the routed distribution is the binding axis and the mechanism claim from #2 is
  completed; if it still loses, the −2.3% ppl win needs a different (non-reconstruction) story.

## Arm B — router drift under quantization

Fake-quantize the full model at 2.0 bpw (uniform, the shipped operating point) and ask whether the
router changes its top-8 selections on the *same* tokens.

- Two identical forward passes over 512 C4 rows × 2048 tokens: fp16, then after
  `quantize_experts(lm, avg_bits=2.0)` (in-place). A router hook records the top-k index tensor per
  layer (`model.language_model.layers.{i}.mlp.gate` output `[2]`, uint8).
- Metrics per layer, with a **content-change null** (fp16's own selections on a disjoint row half)
  so "quantization changed routing" is read against "routing changes with the text anyway":
  - per-slot selection agreement: fp16-vs-quant on identical tokens, vs fp16-halfA-vs-halfB.
  - selection-frequency L1 and correlation, fp16 vs quant.
  - top-8 hot-set overlap (Jaccard).
- Depth structure: drift should grow with layer depth (layer 0 sees no prior expert error).
- Correlation at layer 13: per-expert output error (Arm A) vs per-expert frequency change — does
  quantization *reallocate traffic away from* badly-quantized experts (self-healing) or *toward*
  them (compounding)?
- **Decision:** drift ≪ content null → routing is robust; MoE quantization ≈ dense quantization plus
  per-expert error, and the study's "MoE-unique" framing softens. Drift ≳ content null → the router
  re-routes under quantization; calibration-stale allocation and compounding error are real, and a
  "keep the router path clean" mitigation is worth designing.

## Protocol

```bash
# on pi-a100-80gb, after pytest -q is green
setsid nohup bash /tmp/run_routing_quant.sh </dev/null \
  >>logs/routing_quant_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
```

One fp16 pass serves both arms (routed inputs for A, fp16 selections for B). 512 C4 rows × 2048
tokens, matching Phase 2. A 32-row smoke run first (hook correctness), then the full run.

## Success / failure

| outcome | meaning | follow-up |
|---|---|---|
| A: allocation wins routed-weighted | #2's ppl sign explained; mechanism complete | fold into final claim |
| A: allocation still loses | reconstruction ≠ ppl; something else drives the win | routed-input probe on the e2e encode |
| B: drift ≪ null | routing robust | MoE-unique penalty is per-expert, not structural |
| B: drift ≳ null | router re-routes | clean-router-path mitigation design |

## Risks

- **Memory:** 35B fp16 on GPU (~75 GB) plus routed-input buffers (≤ 32k × 896 × 4 B × 32 experts ≈
  3.7 GB, CPU-side) plus per-layer selection tensors (335 MB × 2 passes). Fits 80 GB; scoring is
  chunked.
- **Router path name coupling:** hooks match `mlp.gate` / `Experts` by name+class, the same
  coupling `ExpertUsageProfiler` already uses and that held on this box's transformers 5.14.1.
- **Determinism:** same token ids, inference mode, no dropout → fp16 selections are reproducible;
  the null and treatment share content.
- **Proxy transfer:** Arm A is still reconstruction (inputs are real, the metric is not ppl). Arm B
  is a direct router measurement — the part that survives a proxy objection.

## Testing

Probe script, not unit-tested (the diagnose convention). No new library surface — reuses
`lloyd_kmeans`/`assign` (covered), `quantize_experts` (covered), `bits_from_frequency` +
`arithmetic_centered` (covered), `load_causal_lm` (box path).
