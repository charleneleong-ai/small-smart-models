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

(The `encode.py` module docstring claims `(num_experts, d_in, d_out)`. That is wrong and is fixed in
this phase.)

## Design — pre-scaling

Per-dimension weighting is applied by change of variables rather than by rewriting the distance metric.
Scale each input channel by `√w_j` before quantizing, fit in the scaled space, and divide back out on
reconstruction. Because

```
Σ_d (√w_d·x_d − √w_d·c_d)²  ≡  Σ_d w_d(x_d − c_d)²
```

plain k-means on the scaled weights **exactly** minimizes the weighted objective. Pooling scaled
sub-vectors from all groups into one shared codebook still minimizes the true global weighted error —
the substitution is per-element, so the pooled sum is the weighted sum.

The alternative — collapsing each sub-vector's four weights to their mean and running weighted k-means —
is an approximation of the same objective for no gain in simplicity, and is out of scope.

### Where it lives

Entirely in [`quantize_fused_experts`](../../src/smart_quant/encode.py).
[`lloyd_kmeans`](../../src/smart_quant/codebook.py),
[`pq_quantize`](../../src/smart_quant/codebook.py) and
[`pq_dequantize`](../../src/smart_quant/codebook.py) are **untouched**:

```python
w_sqrt = importance[e].sqrt()                                   # (in_features,)
codes, cbs = residual_pq_quantize(weight[e] * w_sqrt, ...)      # fit in scaled space
weight[e] = (pq_dequantize(codes, cbs) / w_sqrt).to(weight.dtype)
```

### `pq_bpw` — scale storage

The scale vector must be reconstructable at inference, so it is stored and counted. `pq_bpw` gains an
optional `scale_len: int | None = None` term adding `scale_len * 16` bits:

| tensor | `w` length | added bpw (per-expert) | added bpw (per-layer) |
|---|---|---|---|
| `gate_up_proj` | 2048 | 2048·16 / (1024·2048) = **0.0156** | /256 → 0.00006 |
| `down_proj` | 512 | 512·16 / (2048·512) = **0.0078** | /256 → 0.00003 |

### Matched-footprint discipline

Weighting does not change `k`, index count, or codebook size — only where centroids land — so the only
footprint delta is the scale vector above. Realized `expert_bpw` for the per-expert arm lands ~0.012
higher than its unweighted pair (≈2.012 against `pq2-uniform`'s 2.000). **This is reported honestly
rather than tuned away**: shrinking `k` to force a digit-exact match would handicap the arm under test
with fewer centroids, which is a worse distortion than a 0.6% footprint difference stated plainly. The
per-layer arm is matched to within 0.0001 and needs no caveat.

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

A zero-token expert resolves exactly to `w_layer`. Each `w` is then normalized to mean 1, which also
keeps `√w` near unity so the scaled weights stay in a sane numeric range.

`α` (applying `w^α`) is exposed but defaults to `1.0`. Unmoderated activation magnitudes span orders of
magnitude and can make k-means degenerate, with a few hot rows capturing every centroid; `α<1` is the
lever if the pure version degenerates rather than merely losing. Default stays pure so the first answer
is clean.

## Wiring

- [`encode.py`](../../src/smart_quant/encode.py) — `quantize_fused_experts` gains `sample_weight`
  (`(in_,)` shared or `(num_experts, in_)` per expert) and does the scale/unscale; `quantize_experts`
  gains `importance: dict[str, torch.Tensor] | None` keyed by full fused parameter name, since
  `gate_up_proj` and `down_proj` have different `in_features`. Absent weights take the existing path,
  so Phase-5/6 encodes stay byte-identical.
- [`cli.py`](../../src/smart_quant/cli.py) — `--importance-path` and `--importance-granularity` on
  `encode-eval`, plus a `profile-activations` command. Labels are `wpq<bpw>-<granularity>`.

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

- **Regression safety** — `sample_weight=None` leaves `quantize_fused_experts` byte-identical to today.
- **Weighting has the intended effect** — heavily-weighted input channels reconstruct measurably better
  than under the unweighted fit, and unweighted channels correspondingly worse.
- **Exactness of the change of variables** — on a small case, the scaled-space fit achieves lower
  *weighted* error than the unweighted fit, confirming the objective actually being minimized.
- **`pq_bpw` scale term** — `scale_len` adds exactly `scale_len·16` bits and matches the table above.
- **Shrinkage** — zero-token expert resolves exactly to the layer statistic; high-count expert
  approaches its raw statistic; every row normalized to mean 1.
- **Profiler** — attributes per routed expert from `top_k_index`, and the `down_proj` recompute matches
  a hand-computed `act_fn(gate)*up` on a tiny fixture.

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

- **Per-sub-vector scalar weighting** — an approximation of the same objective; only interesting if the
  scale-vector storage ever becomes a real constraint, which at 0.0156 bpw it is not.
- **Phase 6b vptq spike** — still deferred, not cancelled.
- **AQLM joint beam-search** — unchanged from Phase 6a.
- **`.pre-commit-config.yaml` ruff C901/PLR0915 gate** — repo-wide follow-up.
