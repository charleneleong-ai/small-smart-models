# Phase 7 — activation-weighted first-order PQ

**Study:** bits-per-brain · **Branch:** `feat/weighted-pq-phase7` (off `main`, PRs #6 + #7 merged)
**Date:** 2026-07-29 · **Status:** design approved; mechanism revised 2026-07-29 after box recon

## Problem

Phase 6a found that a second-order residual codebook loses to first-order PQ at matched footprint, and
concluded the cause was budget starvation — splitting one index budget across two stages leaves each
stage too coarse. That conclusion has a confound: [`residual_pq_quantize`](../../src/smart_quant/codebook.py)
is greedy *and unweighted*. Its [`lloyd_kmeans`](../../src/smart_quant/codebook.py) minimizes raw weight
MSE, treating every weight as equally important. Real VPTQ is Hessian-weighted — it spends resolution
where the loss actually cares. So Phase 6a may have tested "naive residual VQ", not "residual VQ".

Phase 7 isolates that variable. It applies activation-derived importance weighting to the **first-order**
quantizer, with no residual stages at all: does spending codebook resolution where activations are large
beat uniform PQ at matched footprint?

## The reframe (why this, not the Phase 6b vptq spike)

Phase 6b was scoped as a time-boxed spike on whether `vptq>=0.0.5` can plumb `qwen3_5_moe` experts.
It carries the architecture-support risk that already killed GGUF and AWQ in Phase 3, and if it is
blocked it returns no signal about the underlying question.

The importance weighting is the part of VPTQ worth having, it is orthogonal to residual stages, and it
is arch-agnostic like the rest of this toolkit. Testing it on the first-order quantizer avoids both the
vptq plumbing risk and Phase 6a's granularity floor. Phase 6b stays deferred, not cancelled.

## The layout fact this design turns on

Confirmed on the box — see [recon findings](../plans/2026-07-29-phase7-recon-findings.md). Fused expert
weights are `(num_experts, out_features, in_features)`, the standard `nn.Linear` layout, because the
forward uses `F.linear(x, W)` = `x @ W.T`:

```
gate_up_proj (256, 1024, 2048)     # out = 2 x moe_inter, in = hidden
down_proj    (256, 2048, 512)      # out = hidden,        in = moe_inter
```

So `out, in_ = weight[e].shape` in [`quantize_fused_experts`](../../src/smart_quant/encode.py) is
correctly named, and `pq_quantize` splits `in_` into `sub_dim`-wide groups — **sub-vectors run along the
input dim**. Importance `E[x_j²]` is per input channel, so the four elements of a sub-vector carry four
*different* weights. The weighting is per-dimension, not per-sample.

(The `encode.py` module docstring claimed `(num_experts, d_in, d_out)` — wrong, and corrected in this
phase. The code was always right.)

## Design — per-dimension weighted k-means

The codebook is fit by minimizing the weighted objective directly. Assignment and centroid update both
use the per-dimension weighted squared distance, expanded to avoid materializing an `(n, k, d)`
difference:

```
||x − c||²_w  =  Σ_d w_d x_d²  −  2 Σ_d w_d x_d c_d  +  Σ_d w_d c_d²
```

Three matmuls — the same asymptotic cost as the `cdist` it replaces, since that is already
`(out·groups) × k × d`. Both steps route through one shared
[`assign`](../../src/smart_quant/codebook.py) helper so the fit and the final assignment cannot diverge.

**Why not pre-scaling.** Scaling columns by `√w`, fitting, and dividing back out is exactly equivalent
for the objective — but over a *different reconstruction family*. It gives group `g` the effective
codebook `C/√w_g`, where the unweighted baseline uses one `C` for every group. Neither family contains
the other, so "minimizes the objective over its own family" says nothing about beating the baseline —
and measured on this quantizer it loses (0/5 seeds i.i.d., 4/5 structured, against 5/5 for weighted
k-means). The cause is that scaling spreads the pooled sub-vectors over a wider region, so a *shared*
codebook — the thing that makes ~2 bpw reachable — fits everything worse. Pre-scaling also stores `w`
and confounds weighting with AWQ-style per-channel rescaling. See
[Out of scope](#out-of-scope).

### Where it lives

[`lloyd_kmeans`](../../src/smart_quant/codebook.py) gains `dim_weight (n, d)`;
[`pq_quantize`](../../src/smart_quant/codebook.py) and
[`residual_pq_quantize`](../../src/smart_quant/codebook.py) gain `channel_weight (in_,)` and tile it to
the pool. [`pq_dequantize`](../../src/smart_quant/codebook.py) and
[`pq_bpw`](../../src/smart_quant/codebook.py) are **untouched** — reconstruction is still a plain
codebook lookup.

The one subtlety is the `max_fit` subsample: points and weights must be indexed by the *same* tensor,
so the selection is hoisted to a named `sel` rather than inlined twice.

### Matched-footprint discipline

Weights steer the fit and nothing else — `k`, index count, and codebook size are all unchanged, and
nothing extra is stored. Realized `expert_bpw` for a weighted encode is therefore **identical** to its
unweighted pair (`wpq20-*` at 2.000 against `pq2-uniform`'s 2.000, `wpq25-*` at 2.542 against
`pq25-uniform`'s 2.542). No budget search and no caveat — the cleanest matched comparison in the study.

### What weighting can and cannot exploit

Weighting only pays when input channels genuinely differ. On i.i.d. columns every sub-vector is
exchangeable, so favouring some merely wastes centroids — verified: the advantage vanishes entirely on
`randn` data and is a consistent ~11% on heterogeneous columns, holding at every subsample ratio down to
1/16. Unit tests therefore fit heterogeneous matrices; an i.i.d. fixture would test the one regime where
the method is expected to do nothing.

## Calibration

Both projections get real statistics — recon verdict **FULL**. The `Experts.forward` signature is

```
forward(hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) -> Tensor
```

so a single `forward_pre_hook` receives the hidden states **and** the routing. There is no router hook,
no pairing by layer, and no hook-ordering hazard. `hidden_states` arrives already flattened to
`(tokens, hidden)`; `top_k_index` is `(tokens, top_k)`.

`ActivationImportanceProfiler` in [`expert_importance.py`](../../src/smart_quant/expert_importance.py):

- Hooks each `Experts` module, accumulating `Σx²` and a token count per (tensor, expert).
- `gate_up_proj`: statistic is over `hidden_states` directly.
- `down_proj`: the intermediate is recomputed inside the hook from the captured input, using the same
  packing the forward uses — `gate, up = F.linear(x_e, gate_up_proj[e]).chunk(2, -1)`, then
  `act_fn(gate) * up`.
- Retains only running sums, never raw activations, so memory is bounded at
  `n_layers × n_experts × d_in` floats held on CPU.
- Caches to `expert_act_importance.pt` beside the existing `expert_freq.pt` (untracked box cache).

### Cold-expert shrinkage

At top-8 of 256, tail experts see few tokens, so raw per-expert `E[x²]` is noisy — the failure mode
where the per-expert arm loses for a reason unrelated to the hypothesis. Shrink toward the layer
statistic by token count:

```
w_e = (n_e · w_e_raw + τ · w_layer) / (n_e + τ)      # τ = pseudo-count, default 1000 tokens
```

A zero-token expert resolves exactly to `w_layer`. Each `w` is then normalized to mean 1, so the fit is
invariant to the statistic's absolute scale and comparable across experts.

`α` (applying `w^α`) is exposed but defaults to `1.0`. Unmoderated activation magnitudes span orders of
magnitude and can make k-means degenerate, with a few hot channels capturing every centroid; `α<1` is the
lever if the pure version degenerates rather than merely losing. Default stays pure so the first answer
is clean.

## Wiring

- [`encode.py`](../../src/smart_quant/encode.py) — `quantize_fused_experts` gains `channel_weight`
  (`(in_,)` shared or `(num_experts, in_)` per expert); `quantize_experts`
  gains `importance: dict[str, torch.Tensor] | None` keyed by full fused parameter name, since
  `gate_up_proj` and `down_proj` have different `in_features`. Absent weights take the existing path,
  so Phase-5/6 encodes stay byte-identical.
- [`cli.py`](../../src/smart_quant/cli.py) — `--importance-path` and `--importance-granularity` on
  `encode-eval`, plus a `profile-activations` command. Labels are `wpq<bpw>-<granularity>`.

Note `quantize_experts` keys `importance` by **full parameter name**, not layer: `gate_up_proj` and
`down_proj` have different `in_features` (2048 vs 512), so a per-layer key cannot serve both.

## Experiment

Three sequential encodes on `pi-a100-80gb` (only one fp16 model fits at a time), as detached PPID=1
daemons, plus two forward-only calibration passes:

| label | granularity | target | pairs against | reference ppl |
|---|---|---|---|---|
| `wpq20-expert` | per-expert | ~2.0 | `pq2-uniform` (2.00) | 6.77 |
| `wpq25-expert` | per-expert | ~2.5 | `pq25-uniform` (2.54) | 6.21 |
| `wpq25-layer` | per-layer | ~2.5 | `wpq25-expert`, `pq25-uniform` | — |

Per-expert takes both footprints; per-layer takes one, at 2.5, to disambiguate a per-expert loss.
Rows append to `experiments/bits-per-brain/results.jsonl` (gitignored, regenerated).

All four outcomes are informative:

| per-expert | per-layer @2.5 | reading |
|---|---|---|
| wins | — | specialization-aware weighting works; margin over imatrix widens past 3.2 pp |
| loses | wins | weighting works, per-expert estimates too noisy — raise τ or extend calibration |
| loses | loses | weighting does not help this quantizer |
| wins | loses | weighting pays only with specialization — strongest form of the hypothesis |

A double negative is a publishable verdict. With Phase 3 (expert-level bit allocation counterproductive)
and Phase 6a (residual VQ loses at matched footprint), it would establish that this quantizer resists
every form of non-uniform allocation tried against it and uniform first-order PQ keeps winning.

## Testing

New cases in [`tests/test_encode.py`](../../tests/test_encode.py),
[`tests/test_codebook.py`](../../tests/test_codebook.py), and
[`tests/test_expert_importance.py`](../../tests/test_expert_importance.py), by area:

- **Regression safety** — absent weights leave `lloyd_kmeans`, `pq_quantize` and
  `quantize_fused_experts` byte-identical to today; an all-ones `dim_weight` agrees with the unweighted
  fit, which `None` alone would not catch.
- **Optimizes what it claims** — weighted k-means achieves lower *weighted* error than the unweighted
  fit, and the heavier dimension of a synthetic pair is fit more closely.
- **`max_fit` weight alignment** — reversing the weight vector is the misalignment control: had the
  subsample paired weights with the wrong points, the true and reversed vectors would be equally
  (un)helpful.
- **Footprint invariance** — `quantize_fused_experts` returns identical `(realized_bits, n_weights)`
  with and without weights.
- **Shrinkage** — zero-token expert resolves exactly to the layer statistic; high-count expert
  approaches its raw statistic; every row normalized to mean 1.
- **Profiler** — attributes per routed expert from `top_k_index`, and the `down_proj` recompute matches
  a hand-computed `act_fn(gate)*up` on a tiny fixture.

All quantizer fixtures use **heterogeneous** columns for the reason given above.

## Plot & doc

- [`experiments/plot_quality_vs_bpw.py`](../../experiments/plot_quality_vs_bpw.py) — add
  `weighted_curve` selecting `label.startswith("wpq")`. The existing
  [`uniform_curve`](../../experiments/plot_quality_vs_bpw.py) (`endswith("-uniform")`, not `rvq`) and
  [`residual_curve`](../../experiments/plot_quality_vs_bpw.py) (`startswith("rvq")`) both miss the
  `wpq*-{expert,layer}` labels, so the new curve is additive and cannot pollute the Phase-5 line.
- [`docs/experiments/bits-per-brain.md`](../experiments/bits-per-brain.md) — Phase-7 section with
  hypothesis, matched-footprint table, and verdict.

## PR plan

One PR, `feat/weighted-pq-phase7`, off `main`. Calibration profiler + scaled-space quantization +
wiring + tests + encodes + plot + doc.

## Out of scope

- **Pre-scaling (columns × `√w`)** — measured and rejected as the mechanism (above), but it is a real
  technique in its own right: the per-group rescaling it induces is essentially AWQ-style per-channel
  scaling. Worth revisiting as its **own arm** if weighting shows signal — never folded into the
  weighted encode, where a win could not be attributed to weighting versus rescaling.
- **Per-sub-vector scalar weighting** — collapsing each sub-vector's four weights to their mean. Measured
  as near-inert (1/5 and 4/5 seeds, barely distinguishable from the unweighted fit), because a scalar
  weight cannot change the assignment at all.
- **Phase 6b vptq spike** — still deferred, not cancelled.
- **AQLM joint beam-search** — unchanged from Phase 6a.
- **`.pre-commit-config.yaml` ruff C901/PLR0915 gate** — repo-wide follow-up.
