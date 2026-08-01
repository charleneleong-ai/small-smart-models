# Uniform shared-codebook PQ is a strong local optimum

**Date:** 2026-08-01 · **Model:** Qwen3.6-35B-A3B, routed expert FFNs, ~2.0–2.5 bpw

Ten measurements now bear on one question: can anything beat plain uniform product quantization
with a single learned codebook per expert at `sub_dim=4`? Nine say no. The tenth — swapping the
learned codebook for a **lattice** — is the only direction that escapes, and it does so by ~10%.

This note consolidates them, because four scattered negative phases read as bad luck while ten
aligned measurements read as a result.

## The nine that found nothing

| # | what was tried | measured outcome |
|---|---|---|
| 1 | **Expert-level bit allocation** (Ph 3) | `pq2-expert` 7.68 vs `pq2-uniform` 6.77 — badly worse |
| 2 | **Second-order residual codebooks** (Ph 6a) | `rvq25` 6.54 vs `pq25` 6.21; `rvq20` 7.45 vs 6.77 |
| 3 | **Activation weighting** (Ph 7) | `wpq25` 6.2325 vs 6.2137; alpha sweep 0.25→1.0 never reaches baseline |
| 4 | **GPTQ error compensation** (Ph 8) | `gptq25` 6.2400 vs 6.2137 — *actively harmful*, ~3% codebook drift |
| 5 | **Codebook utilization** | 100% outside layer 0; recoverable waste ~0.035 bpw, not the 0.28 first extrapolated |
| 6 | **Harder k-means fitting** | k-means++ 100 iters × 3 restarts buys **1.7–1.8%** |
| 7 | **Data-free column rescaling** | weight columns are homogeneous (p99/p50 1.12–1.54) — nothing to exploit |
| 8 | **Incoherence processing** | mu 6.15→5.03 as advertised, but MSE ±0.5% and the layerwise objective **worse** in 6/6 trials |
| 9 | **Codebook shape** (`sub_dim` × n_books) | shipped `d=4, k=1024, 1 book` wins; `d=2` 8.5% worse; 4 books buys 1.3% for 3.7% more bpw |

Four of these are full end-to-end encodes with perplexity at matched footprint (1–4); the rest are
direct measurements on real expert tensors.

## Why they all fail, in one sentence

Every one of #1–#4 **reallocates capacity inside a fixed structure**, and #5–#9 show that structure
is already close to its own ceiling: the fit is within ~2% of what much harder fitting achieves,
utilization is complete, the shape is optimal among reachable ones, and the two preprocessing tricks
that work elsewhere (rescaling, rotation) have nothing to grip here.

Two results sharpen it further. Phase 7's post-mortem showed `E[x²]` **is** the layerwise Hessian
diagonal and is *anti*-correlated with where PQ errs — so reweighting by any diagonal is closed, not
just that one. Phase 8's drift measurement showed compensation *displaces* weights away from
centroids fit beforehand, costing ~3% fit for whatever it gains. A learned codebook punishes
anything that disturbs it.

## The one exit: lattice quantization

The structural constraint is that a stored codebook costs `O(2^{kd}·d)` to keep *and* to search —
exponential in both bitrate and dimension. Our own shape sweep shows the wall directly: k-means at
`sub_dim=8, k=65536` costs **6.0 bpw, of which 4.0 is codebook storage alone**, and still loses to
`sub_dim=2` at 3.0 bpw.

A lattice escapes it because its points are *computed*, not tabulated. E8 — the densest sphere
packing in 8 dimensions, and what QuIP# uses — makes `sub_dim=8` reachable at zero storage.

Pooled over 32 experts of layer 13 (8.4M sub-vectors, so a ~2^20 shell genuinely binds), fixed-rate,
no entropy coding:

| method | bpw | recon MSE | distinct codes |
|---|---|---|---|
| k-means d=4, k=1024, **per expert** | 2.531 | 6.4031e-06 | 32,768 |
| E8 s=0.0100 | 2.455 | 7.1683e-06 | 817,249 |
| E8 s=0.0080 | 2.605 | 4.5877e-06 | 1,881,900 |

Interpolated to 2.531 bpw, E8 gives **≈5.7–5.9e-06 — about 8–11% better**. A single global lattice,
with no per-expert adaptation at all, beats 32 individually fitted codebooks.

### How that number shrank, and why the final one is the honest one

| framing | apparent gain | what was wrong |
|---|---|---|
| entropy-coded rate, single expert | 2.3× | lattice got ideal variable-length coding; k-means paid fixed `log2(k)` |
| fixed-rate 2^16 shell, single expert | 16% | valid, but only at 2.0 bpw, not our operating point |
| **fixed-rate, pooled, matched 2.53 bpw** | **~10%** | shell finally binds — pool 8.4M ≫ shell 1M |

The intermediate 2^20 single-expert run was degenerate and is discarded: the pool held 262,144
sub-vectors while the shell kept ~1.05M, so nearly every sub-vector received its own code —
memorisation, not quantisation.

## What this does and does not license

**Does:** it identifies the codebook *family* as the binding constraint, and gives a measured reason
to expect a structured quantizer to help where nine allocation and preprocessing tricks did not.
It also predicts, testably, that incoherence processing *would* pay off for a lattice even though it
did not for k-means — a fixed codebook cannot adapt to the data's shape, so reshaping the data to fit
it is exactly the right move.

**Does not:** ~10% reconstruction MSE is not ~10% perplexity. This study has repeatedly watched
reconstruction gains fail to transfer — Phase 7 improved its own objective by 41% and *lost*
perplexity. No encode has been run with a lattice quantizer. The measurement is one layer, 32
experts, `gate_up_proj` only, and a practical implementation still needs canonical enumeration of a
~2M-point shell and an efficient encoder.

Reproduce with `experiments/diagnose_lattice.py`.
