# Uniform shared-codebook PQ is a strong local optimum — with one measured exception

**Date:** 2026-08-01 · **Model:** Qwen3.6-35B-A3B, routed expert FFNs, ~2.0–2.5 bpw

Twelve measurements bear on one question: can anything beat plain uniform product quantization
with a single learned codebook per expert at `sub_dim=4`? Eleven say no. The tenth — swapping the
learned codebook for a **lattice** — looked like an escape under a reconstruction proxy and was
refuted by the encode that followed. The eleventh closes the direction that one pointed at, before
any of it was built. The twelfth returns to the first and finds its negative was a footprint
artifact: at a genuinely matched footprint, usage allocation is a **small win** — the one measured
exception to the local optimum.

This note consolidates them, because four scattered negative phases read as bad luck while eleven
aligned measurements read as a result.

## The twelve measurements

| # | what was tried | measured outcome |
|---|---|---|
| 1 | **Expert-level bit allocation** (Ph 3) | `pq2-expert` 7.68 vs `pq2-uniform` 6.77 *looked* like a loss; at true matched footprint `pq2-expert-storage` 6.61 is a **−2.3% win** (see #12) |
| 2 | **Second-order residual codebooks** (Ph 6a) | `rvq25` 6.54 vs `pq25` 6.21; `rvq20` 7.45 vs 6.77 |
| 3 | **Activation weighting** (Ph 7) | `wpq25` 6.2325 vs 6.2137; alpha sweep 0.25→1.0 never reaches baseline |
| 4 | **GPTQ error compensation** (Ph 8) | `gptq25` 6.2400 vs 6.2137 — *actively harmful*, ~3% codebook drift |
| 5 | **Codebook utilization** | 100% outside layer 0; recoverable waste ~0.035 bpw, not the 0.28 first extrapolated |
| 6 | **Harder k-means fitting** | k-means++ 100 iters × 3 restarts buys **1.7–1.8%** |
| 7 | **Data-free column rescaling** | weight columns are homogeneous (p99/p50 1.12–1.54) — nothing to exploit |
| 8 | **Incoherence processing** | mu 6.15→5.03 as advertised, but MSE ±0.5% and the layerwise objective **worse** in 6/6 trials |
| 9 | **Codebook shape** (`sub_dim` × n_books) | shipped `d=4, k=1024, 1 book` wins; `d=2` 8.5% worse; 4 books buys 1.3% for 3.7% more bpw |
| 10 | **E8 lattice** (Ph 9) | `e8-25` 6.4607 vs 6.2137; `e8-20` **8.6827** vs 6.765 — loses, and the gap widens as rate falls |
| 11 | **Zero-storage codebook** (Ph 10) | learned-vs-drawn gap **widens** 13.4% → 21.2% as k grows 256 → 16384 — the trellis premise fails |
| 12 | **Allocation mechanism 2×2** | reconstruction probe: water-fill helps scalar (−3.6%), slightly hurts PQ (+1.0%); the corrected encode flips PQ to a **−2.3% ppl win** — Phase 3's negative was footprint end to end |

Six of these are full end-to-end encodes with perplexity (1–4, 10, plus the corrected
`pq2-expert-storage` row); the rest are direct measurements on real expert tensors.

### #12 corrected the first measurement

The 2×2 ([`experiments/diagnose_allocation_2x2.py`](experiments/diagnose_allocation_2x2.py),
[writeup](allocation-2x2-diagnosis.md)) ran the usage water-fill against both quantizer families
on real layer-13 weights. Two findings. First, `bits_from_frequency` pins the *usage-weighted* mean,
which is not the storage cost — with real routing skew the aggressive Phase-3 allocation realized
**1.63 bpw**, not 2.0, so `7.68 vs 6.77` was a smaller-tensor score, and the
"monotone in allocation strength" pattern is exactly what footprint drift produces. Second, with a
true arithmetic-footprint-matched allocation the reconstruction probe shows the predicted
mechanism — the water-fill helps a rate-limited scalar family (−3.6%) and slightly hurts the
fit-limited PQ (+1.0%).

The corrected end-to-end encode resolved the tension. Re-centred so the *storage* mean is 2.0 bpw,
`pq2-expert-storage` scores **6.607 vs 6.765** at ~2.099 realized bpw — allocation is a free win,
and the confounded 7.68/7.07 were entirely footprint. The probe's PQ sign failed to transfer
because it sampled rows uniformly, which cannot see that ppl rewards the *routed* experts. The
field's allocations help because they hold their byte budget fixed; the study's original water-fill
did not.

## Why they all fail, in one sentence

Every one of #2–#4 **reallocates capacity inside a fixed structure**, and #5–#9 show that structure
is already close to its own ceiling: the fit is within ~2% of what much harder fitting achieves,
utilization is complete, the shape is optimal among reachable ones, and the two preprocessing tricks
that work elsewhere (rescaling, rotation) have nothing to grip here. #1 — the one that reallocates
*across* experts rather than within a tensor — is the exception, and only once its byte budget was
held fixed (see #12).

Two results sharpen it further. Phase 7's post-mortem showed `E[x²]` **is** the layerwise Hessian
diagonal and is *anti*-correlated with where PQ errs — so reweighting by any diagonal is closed, not
just that one. Phase 8's drift measurement showed compensation *displaces* weights away from
centroids fit beforehand, costing ~3% fit for whatever it gains. A learned codebook punishes
anything that disturbs it.

## The apparent exit that wasn't: lattice quantization

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

Interpolated to 2.531 bpw, E8 gives ≈5.7–5.9e-06 — apparently 8–11% better, from a single global
lattice with no per-expert adaptation at all.

**Phase 9 built it and the encode refuted it**, in the direction the caveat below anticipated:

| footprint | uniform PQ | E8 | delta |
|---|---|---|---|
| 2.5 bpw | 6.2137 | 6.4607 | −0.247 |
| 2.0 bpw | 6.765 | **8.6827** | **−1.918** |

The gap *widening* as the rate falls is the diagnostic: that is the signature of missing shape gain,
not of a fixable implementation detail. The proxy's ~10% was also interpolation error — the
rate-distortion curve swings 56% over 0.15 bpw, and measured at exactly 2.500 the lattice is ~8%
*worse*, not better.

### The prediction about rotation was wrong too

The version of this note that preceded Phase 9 predicted that incoherence processing *would* pay off
for a lattice even though it did not for k-means. Measured, a random-sign Hadamard buys **0.5% at
2.5 bpw and 1.4% at 2.0** — against gaps of 8% and 49%. Incoherence processing makes a distribution
*isotropic*; it does not make it *uniform*. Weights stay Gaussian-peaked after rotation, and a
lattice's cells are uniform-density, so the mismatch that actually costs the lattice survives.

### What the shrinking number was really tracking

| framing | apparent gain | what changed |
|---|---|---|
| entropy-coded rate, single expert | 2.3× | lattice got ideal variable-length coding; k-means paid fixed `log2(k)` |
| fixed-rate 2^16 shell, single expert | 16% | valid, but only at 2.0 bpw, not our operating point |
| fixed-rate, pooled, matched 2.53 bpw | ~10% | shell binds — pool 8.4M ≫ shell 1M |
| **end-to-end encode, matched bpw** | **−4% to −28%** | perplexity, not reconstruction |

Each step was framed at the time as removing a confound. In hindsight the first step removed the
*mechanism*: a lattice's competitiveness depends on entropy coding, because its codes are uniform in
space and therefore highly non-uniform in frequency. Fixed-width indexing spends equal bits on
rarely-occupied outer shells. k-means codes are already near-uniform in frequency (measured entropy
9.90–9.93 of 10 bits), so the same coding buys them nothing — which is exactly why the comparison
looked lopsided when only one side had it.

The intermediate 2^20 single-expert run was separately degenerate and is discarded: the pool held
262,144 sub-vectors while the shell kept ~1.05M, so nearly every sub-vector received its own code —
memorisation, not quantisation. `calibrate_scale` now raises rather than return such a fit.

## The trellis exit, measured before building it

Phase 9's closing direction was a trellis. QTIP's bet is that one buys an enormous *effective*
codebook at zero storage: codewords are pseudorandomly generated from a Gaussian rather than
tabulated, making the codebook simultaneously storage-free and shape-matched. The cheap form of that
question, asked before writing a Viterbi decoder — at identical index cost, how much does *learning*
where codewords sit buy over *drawing* them from the right distribution, and does it shrink as k
grows?

Four codebooks at identical bits per index, layer 13, `d=4`, 8 experts. Each expert's 524,288
sub-vectors are split by parity into disjoint halves — codebooks are fit on 262,144 from one half
and scored on 131,072 from the other, so the two k-means rows are never scored in-sample
(relative reconstruction error, lower is better; `gauss gap` is `gaussian-global` over
`kmeans-expert`):

| k | bpw | kmeans-expert | kmeans-global | gaussian-global | d4-global | gauss gap |
|---|-----|---------------|---------------|-----------------|-----------|-----------|
| 256 | 2.00 | 0.3160 | 0.3173 | 0.3583 | 0.5253 | 13.4% |
| 1024 | 2.50 | 0.2256 | 0.2288 | 0.2553 | 0.3213 | 13.2% |
| 4096 | 3.00 | 0.1585 | 0.1658 | 0.1823 | 0.1924 | 15.0% |
| 16384 | 3.50 | 0.1070 | 0.1195 | 0.1297 | 0.1361 | 21.2% |

The last column decides it. If the trellis's huge effective k were the missing piece, the gap would
close as k grows. It widens — 13.4% → 21.2%. More effective codebook moves a shape-matched free
codebook *further* from a learned one, not closer, so the mechanism QTIP relies on does not hold on
these weights.

Splitting the gap into its two axes shows why nothing zero-storage recovers it. At 2.00 bpw,
per-expert *adaptation* is worth 0.4% (`kmeans-expert` → `kmeans-global`) while *learned placement*
is worth 12.9% (`kmeans-global` → `gaussian-global`) — Lloyd's spreading does nearly all the work,
and drawing from the correct distribution does not reproduce it. By 3.50 bpw adaptation becomes the
larger term (11.7% vs 8.5%). A generated codebook can have neither.

Two side results fall out. `d4-global` — spread optimally, shaped wrong — is 46.6% worse than
`gaussian-global` at k=256 but only 4.9% worse at k=16384, reproducing Phase 9's rate-dependent
lattice penalty from an independent instrument. And at the rates we ship, one *global* codebook
costs only 0.4–1.4% over 32 per-expert ones, which is a simplification available for the taking
rather than another null.

Two limits on the claim. This is reconstruction error, not perplexity, and Phase 7 stands as the
reminder that the two can disagree — here the proxy argues *against* the trellis, so the risk is
that it understates one, though a gap widening 13% → 21% makes a reversal unlikely. And
`gaussian-global` models a zero-storage, shape-matched, unlearned codebook, not trellis *sequence*
decoding; what is refuted is the codebook-shape premise, not every conceivable trellis.

## What this leaves

The codebook storage wall is real, and both structured ways around it are now measured: a fixed-rate
lattice loses on shape (Phase 9), and a generated zero-storage codebook loses on placement, by more
as the rate rises (Phase 10). Entropy coding is the one mechanism still untested — it is what made
the lattice look competitive before that confound was removed — but a learned codebook's codes are
already near-uniform in frequency, so it is a route to making *lattices* viable rather than to
beating what we ship.

Reproduce with `experiments/diagnose_lattice.py` (proxy), `experiments/diagnose_rotation.py`
(rotation), `experiments/diagnose_codebook_shape.py` (shape vs placement), and the `e8-20` /
`e8-25` rows of `results.jsonl` (encodes).
