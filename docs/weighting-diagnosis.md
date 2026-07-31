# Why activation weighting failed — Phase-7 post-mortem

**Study:** bits-per-brain · **Date:** 2026-07-30 · **Follows:** Phase 7 ([#8](https://github.com/charleneleong-ai/small-smart-models/pull/8))
**Reproduce:** `PYTHONPATH=src python experiments/diagnose_weighting.py --importance <path.pt>`

Phase 7 measured that activation-weighted PQ loses to uniform PQ at matched footprint, and closed
the two obvious escape hatches (noisy per-expert statistics; outlier degeneracy). It did not explain
*why*. The end-to-end encodes cannot: perplexity is one number at the end of a long pipeline.

This note measures the two things that distinguish "capacity reallocation cannot work" from
"we reallocated by the wrong signal", on real expert tensors at the real encode settings.

## Method

`gate_up_proj`, layers 0 / 13 / 26, experts 0 / 128, k=1024 (2.5 bpw), iters=10,
`max_fit=8192` — matching [`quantize_fused_experts`](../src/smart_quant/encode.py) exactly.
Importance is the per-expert artifact from the Phase-7 calibration pass (512 C4 rows).

| layer | exp | codebook util | corr(imp, err) | top-1% err | bottom-50% err | wMSE gain | w-err in top-1% |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 46.0% | **+0.310** | −63.2% | +34.2% | 40.8% | 61.0% |
| 0 | 128 | 69.5% | −0.179 | −92.9% | +39.5% | 51.8% | 48.9% |
| 13 | 0 | 100% | −0.157 | −80.2% | +11.9% | 7.4% | 9.6% |
| 13 | 128 | 100% | −0.170 | −82.6% | +12.9% | 9.7% | 11.5% |
| 26 | 0 | 100% | −0.206 | −85.3% | +18.3% | 14.3% | 14.7% |
| 26 | 128 | 100% | −0.041 | −81.4% | +19.6% | 13.5% | 14.9% |

## The mechanism works

Top-1%-importance channels reconstruct 63–93% better under weighting, in every tensor sampled.
Weighted k-means does exactly what it was built to do; nothing here is a plumbing failure, and the
Phase-7 result is not an implementation artifact.

## `E[x²]` is the wrong signal

`corr(importance, unweighted per-channel error)` is **negative in five of six tensors** (−0.04 to
−0.21). High-activation channels are not where the quantizer is struggling — if anything they are
marginally easier. So weighting spends resolution repairing channels that were not broken, and the
bulk pays for it (+12–20% on the bottom half).

That reframes the Phase-7 verdict. The finding is **not** that non-uniform capacity allocation
cannot beat uniform PQ. It is that this particular signal points away from the error.

### `E[x²]` *is* the layerwise Hessian diagonal

Worth stating explicitly, because the obvious next move — "use real curvature instead of activation
magnitude" — is not a next move at all. For the layerwise objective `L = E||Wx - West x||^2`, with
`D = W - West`:

```
L = tr(D · E[x x^T] · D^T)        =>        d^2 L / d D_ij^2 = 2 · E[x_j^2]
```

The diagonal Hessian **is** `2·E[x_j²]`, which is exactly what
[`ActivationImportanceProfiler`](../src/smart_quant/expert_importance.py) accumulates
(`sum_t x_jt^2` = `diag(X X^T)`, by construction). So Phase 7 did not test a crude proxy for
curvature — it tested curvature itself, correctly.

The finding is therefore stronger than "wrong proxy": **curvature of the layerwise reconstruction
objective is not aligned with where product quantization actually errs.** Reweighting the fit by the
diagonal is a dead end, and no better diagonal exists to substitute.

Two directions survive, and neither is a reweighting:

- **Off-diagonal / error compensation** (what GPTQ and VPTQ actually do). Quantize columns
  sequentially and compensate the *remaining* weights for error already committed, using the full
  `E[xx^T]` rather than its diagonal. No new statistic is needed — the calibration hook already sees
  `X`. This is where those methods' advantage lives, and it is untouched by anything above.
- **True end-loss Fisher.** Sensitivity of the LM loss rather than of layerwise reconstruction.
  A genuinely different objective, since low layerwise error does not imply low end loss. Needs
  backprop over the calibration set, so it is a new profiler rather than a new flag.

## Codebook utilization is fine — a retracted claim

An early reading of layer 0 / expert 0 showed 46% codebook utilization and was extrapolated to
"~0.28 bpw of pure waste model-wide", which would have dwarfed every effect this study has chased.
A breadth check across layers 0 / 13 / 26 / 39 × experts 0 / 128 × both projections × {2.0, 2.5} bpw
refuted it: **24 of 32 fits sit at 100% utilization** with code entropy 9.90–9.93 of 10 bits.

Corrected model-wide estimate at 2.5 bpw: `(1 × 0.563 + 39 × 0.021) / 40` ≈ **0.035 bpw** — an order
of magnitude smaller, and comparable to overheads the study already treats as negligible. Not worth
a phase. Recorded here because the retraction is the useful part: the original figure came from a
single tensor that turned out to be the worst case in the sample.

## Layer 0 is atypical

It is the only layer showing codebook under-utilization (46–70% on `gate_up_proj` against 100%
everywhere else), the only tensor with positive `corr(imp, err)`, and its weighted error is ~4×
more concentrated (49–61% in the top 1% of channels, against 9.6–14.9% elsewhere). Any future
result measured on layer 0 alone should be treated as unrepresentative until checked against a
middle layer — this note's own retraction is the cautionary case.

## Caveats

Six tensors, `gate_up_proj` only, one expert pair per layer. Enough to refute the utilization
extrapolation and to establish the sign of `corr(imp, err)` consistently, not enough to quantify
either precisely. `down_proj` was covered in the utilization sweep but not the attribution.
