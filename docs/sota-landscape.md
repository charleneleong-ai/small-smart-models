# Where this study sits — 2026 low-bit quantization landscape

**Date:** 2026-08-01 · **Scope:** post-training weight quantization at ~2–3 bpw, the regime the
bits-per-brain study operates in.

Written after four consecutive negative results ([Phase 3](experiments/bits-per-brain.md),
[6a](experiments/bits-per-brain.md), [7](https://github.com/charleneleong-ai/small-smart-models/pull/8),
[8](https://github.com/charleneleong-ai/small-smart-models/pull/10)), to check whether the study is
attacking the problem the field considers load-bearing. Short answer: partly not.

## The gap: we do no incoherence processing

Every method that made 2-bit usable begins by making the weight matrix *incoherent* — rotating it so
that energy is spread rather than concentrated in a few directions, which is what makes a small
codebook able to cover it. QuIP introduced random-Hadamard incoherence processing; QuaRot, SpinQuant
(learned rotations via Cayley SGD) and ButterflyQuant (learned Givens angles, O(n log n)) refined it.
The consensus framing is blunt: incoherence processing is what *enabled* the first usable 2-bit LLMs.

**This study has no rotation step.** Phases 3, 6a, 7 and 8 all optimize *allocation* — which experts
get bits, how codebooks are staged, how the fit is weighted, how error propagates — on a raw,
un-rotated weight matrix. That does not invalidate the four negatives, but it reframes them: they are
second-order questions asked while the first-order step is missing.

That was the obvious top recommendation. It was then measured, and it does not survive.

### …but it does not transfer to a *learned* codebook

Quantizing `W` directly against quantizing `W@Q` under a random-sign Hadamard `Q` and mapping back
with `Qᵀ` (orthogonal, so the layer's function is unchanged and the footprint is exactly matched),
three sign draws per tensor:

| tensor | mu plain → rotated | recon MSE | layerwise objective |
|---|---|---|---|
| L13 E18 | 6.15 → **5.03** | −0.6 to −0.8% (better) | **+2 to +3% (worse)** |
| L26 E109 | 5.56 → **5.12** | +0.5 to −0.4% (noise) | **+4 to +5% (worse)** |

The rotation does what it claims — incoherence drops materially. But weight-space MSE barely moves,
and the layerwise objective gets **consistently worse across all six trials**.

The likely reason is specific to this study: incoherence processing pays off for **fixed** codebooks —
lattices (QuIP#), trellises (QTIP) — which cannot adapt to the data's shape, so you reshape the data
to fit them. Our codebook is **learned by k-means**; it already adapts to whatever distribution the
weights have. Rotating toward isotropy destroys structure the learned codebook was exploiting, and
spreads error evenly across input directions when the input covariance is markedly low-rank (29–35%
effective rank) — putting more error where the inputs actually live.

If that reading is right, incoherence processing is not a missing first step for us; it is coupled to
the fixed-codebook family and does not carry over. Caveats: this is a **one-sided** rotation on the
input dim where QuIP is two-sided, it is two tensors, and Phase 7 showed the layerwise objective is
itself an unreliable perplexity proxy. Reproduce with
`experiments/diagnose_rotation.py`.

## The exponential constraint, and how the field escapes it

A codebook over `d`-dimensional sub-vectors at `k` bits/dim costs `O(2^{kd}·d)` to store *and* to
search — exponential in **both** bitrate and dimension. Our `(sub_dim, k)` trade-off is exactly this
constraint: at 2.5 bpw, `sub_dim=4` needs k=1024 (65 Kbit/codebook), `sub_dim=6` needs k=32768
(3.1 Mbit — ~60% overhead against the index storage, infeasible).

Essentially every serious method is a different answer to *"how do we get high effective dimension
without an exponentially large codebook?"*:

| method | escape route |
|---|---|
| QuIP# | E8 lattice — optimal 8-D sphere packing, structure instead of storage |
| AQLM | additive multi-codebook, sum of entries from several small books, beam search |
| VPTQ | second-order residual codebook |
| **QTIP** | **trellis coding in 256 dims — *computational* codebooks, no storage overhead at all** |
| CCQ | convolutional codes |

QTIP is the current quality-per-bit reference, and reportedly beats QuIP#, AQLM and GPTVQ; QTIP
without fine-tuning matches or exceeds QuIP#/AQLM *with* it.

Our own contribution here — shared-codebook PQ with honest realized-bpw accounting — is a valid
low-overhead point on this map, but it is the unstructured end of it. Two of our four negatives
(Phase 6a residual staging, Phase 8 compensation) are attempts at exactly what VPTQ and GPTQ already
formalize.

## Our `sub_dim` question has a published answer

FASQ (arXiv 2605.04084) is a calibration-free PQ method for LLM weights and is the closest existing
sub-vector-dimension × codebook-cardinality ablation. An RVQ KV-cache study (arXiv 2410.15704)
independently reports that *smaller* sub-vector dimensions outperform larger ones, and that codebook
**count** matters more than codebook **size**. Both point away from our `sub_dim=4` toward smaller
sub-vectors with more books — which is also the untested "intermediate codebook sharing" lever
(today the code is binary: one shared book, or one per group).

## A real tension with the MoE line

MoQE (arXiv 2310.02410) established that MoE expert layers are unusually robust to quantization —
2-bit experts can outperform a dense model trained on the same data. The field's dominant MoE thread
since has been **expert-wise mixed precision**: MC-MoE, MxMoE, GEMQ, and in 2026 BitsMoE
(spectral-energy-guided allocation, arXiv 2606.00079), AlphaQ (calibration-free, arXiv 2606.04980),
and generalization-guaranteed allocation (arXiv 2604.06515).

**Phase 3 of this study found expert-level bit allocation counterproductive**, and Phases 7 and 8
found two further allocation schemes counterproductive. That is a direct disagreement with an active
line of work. Two readings, both worth stating:

- our setting differs materially (Qwen3.6-35B-A3B, 256 experts top-8, ~2.5 bpw, shared-codebook PQ
  rather than scalar/GPTQ quantization), or
- non-uniform allocation is worth less than the field assumes once the quantizer is a *shared*
  codebook, because reallocation disturbs a fit that is already near-saturated.

The second is what our measurements point at: harder codebook fitting buys only ~1.8%, and Phase 8's
compensation *hurt* by displacing weights ~3% away from centroids fit beforehand. It is a claim worth
making explicitly rather than leaving as four unexplained negatives.

## 1-bit is a QAT story, orthogonal to this work

BitNet b1.58 requires training from scratch; BitNet Distillation (ICLR 2026) fine-tunes pretrained
models down to 1.58-bit. On the PTQ side PTQ1.61 pushes the limit (PB-LLM and BiLLM both drift back
to ~2 effective bits), BTC-LLM reaches 0.7-bit with learnable transforms plus a binary codebook, and
ParetoQ gives scaling laws across 1 / 1.58 / 2 / 4-bit. Field consensus: 4-bit PTQ is solved, and
1.58-bit only pays if you can afford QAT or need edge deployment. Our PTQ-at-2.5-bpw work does not
compete here and should not claim to.

## Implications, ranked

1. **Entropy-coded or trellis quantization.** The one direction still open, and the one the field
   converged on. Phase 9 showed a *fixed-rate* lattice loses outright; the mechanism it was missing
   is entropy coding, which prices in the frequency non-uniformity that a structured code creates.
   QTIP's trellis is this idea carried to its conclusion. The cost is a variable-length decoder.
2. **State the MoE disagreement.** Four negatives on non-uniform allocation, against a field trend
   toward more of it, is a finding — but only if written as one.
3. **Benchmark honestly against structured codebooks.** Our shared-codebook PQ is the unstructured
   baseline; QTIP/QuIP# are what a strong 2-bit result looks like in 2026.

**Not recommended,** all three tried and measured:

- **Incoherence processing** on the current quantizer — mu improves as advertised, the objective does not.
- **Smaller `sub_dim` / more codebooks** — two published results point this way, but our own sweep
  disagrees at our operating point: `d=2` is 8.5% worse and four books buy 1.3% for 3.7% more bpw.
- **A fixed-rate lattice**, with or without rotation — see `docs/local-optimum.md`.

## Caveats on the cross-paper numbers

VPTQ's published comparisons ran without CUDA graphs, FlashAttention or `torch.compile`, so its
reported QuIP#/AQLM figures understate them. Cross-paper perplexities in this area are routinely
measured on different sequence lengths, calibration sets and eval harnesses; none of the numbers above
are directly comparable to this study's wikitext-2 figures without re-running.
