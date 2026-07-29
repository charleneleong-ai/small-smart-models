# Phase 7 — activation-weighted first-order PQ

**Study:** bits-per-brain · **Branch:** `feat/weighted-pq-phase7` (off `main`, PRs #6 + #7 merged)
**Date:** 2026-07-29 · **Status:** design approved, pending spec review

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

[`quantize_fused_experts`](../../src/smart_quant/encode.py) slices a fused `(num_experts, d_in, d_out)`
tensor and does `out, in_ = weight[e].shape` — so what PQ calls `in_` is really **d_out**. Sub-vectors
run along the *output* dim.

Importance `E[x_j²]` is per *input* channel, which is axis 0 — the row index. Every element of a
sub-vector therefore shares one scalar weight. This is what makes the design cheap: the weighting is
per-sample, not per-dimension, so it needs no change to the distance metric and no change to
[`pq_dequantize`](../../src/smart_quant/codebook.py).

It also rules out the pre-scaling alternative (scale rows by `√w`, quantize, unscale). That is exact for
the same objective, but it gives row `i` a codebook scaled by `1/√w_i` — a free per-row scale factor on
top of the weighting. A win could not be attributed to weighting rather than rescaling, and the scale
vector would need storing and accounting in `pq_bpw`. Weighted Lloyd's has neither problem.

## Design

### 1. `lloyd_kmeans` — optional `sample_weight`

```python
def lloyd_kmeans(x, k, iters=10, sample_weight=None) -> tuple[torch.Tensor, torch.Tensor]:
```

Assignment is unchanged — nearest centroid is nearest regardless of weight. Only the centroid update is
weighted: `index_add_` accumulates `w.unsqueeze(1) * x` against `w` instead of `x` against ones. Weights
are normalized to mean 1 on entry. `sample_weight=None` takes the existing code path untouched.

### 2. `pq_quantize` — thread the weight through

Gains `sample_weight: torch.Tensor | None` of length `out` (one per input channel), broadcast across
that row's `groups` sub-vectors to match the flattened `pool`. The `max_fit` strided subsample must be
applied to the weights with the *same* index tensor used for the points — misalignment here corrupts the
fit silently, so it is a named test case rather than an assumed invariant.

The `share_codebook=False` path takes the same weights per row; no separate handling.

### 3. `residual_pq_quantize` — forward the weight

It sits in the call chain (`quantize_fused_experts` → `residual_pq_quantize` → `pq_quantize`) even at
`codebook_order=1`, where it runs a single stage, so it must accept `sample_weight` and pass it through.
Weights apply unchanged at every stage: the residual `weight − Σ recon` has the same rows, so the same
per-input-channel importance holds. This costs one parameter and makes weighting compose with Phase 6a's
residual path for free, though no Phase-7 encode uses `order > 1`.

### 4. `pq_bpw` — unchanged

Weighting moves centroids. It does not change `k`, index count, or codebook size, so realized bpw is
identical to the unweighted encode. This is asserted, not assumed (see Testing).

### Matched-footprint discipline

Because `pq_bpw` is invariant, the comparison is matched **by construction** rather than by tuning:
`wpq20-expert` lands at the same realized `expert_bpw` as `pq2-uniform` (2.00) and `wpq25-expert` at the
same as `pq25-uniform` (2.542). No budget search, and no repeat of the Phase-6a granularity floor that
forced `rvq26` to be relabelled `rvq25`.

## Calibration

Two projections need two statistics:

| tensor | `weight[e]` shape | input whose `E[x²]` is needed | reachable |
|---|---|---|---|
| `gate_up_proj` | (hidden, 2·inter) | hidden states entering the MoE block | yes — `Experts` forward-pre hook |
| `down_proj` | (inter, hidden) | post-activation intermediate | not directly — internal to the fused forward |

**Step 0 of implementation is inspecting the real `Experts` module tree and forward signature on
`pi-a100-80gb`.** `transformers` is deliberately absent from the local/CI surface (`pyproject.toml`
keeps it torch-only), so the hook design below is provisional until verified against the model.

`ActivationImportanceProfiler` in [`expert_importance.py`](../../src/smart_quant/expert_importance.py)
mirrors [`ExpertUsageProfiler`](../../src/smart_quant/expert_importance.py) and **reuses its router
predicate** — that `mlp.gate` name-matching is the one piece already proven to survive the transformers-5
`Qwen3MoeTopKRouter` refactor, and is where a fresh implementation would break the same way.

- Hooks the router (per-token expert indices) and the `Experts` module (input hidden states), paired by
  [`layer_index`](../../src/smart_quant/encode.py).
- Accumulates `Σx²` and a token count per (layer, expert). Never retains raw activations, so the
  statistic is bounded at `n_layers × n_experts × d_in` floats (40 × 256 × d_in), held on CPU —
  a few hundred MB at most for the per-expert case, negligible for per-layer.
- `down_proj`: recomputes the intermediate inside the hook from captured input and the still-fp16
  weights (one extra half-FFN matmul over calibration tokens).
  **Fallback if the fused gate/up packing cannot be split cleanly: weight `gate_up_proj` only and hold
  `down_proj` uniform, recorded as a stated limitation of the phase.**
- Caches to `expert_act_importance.pt` beside the existing `expert_freq.pt` (untracked box cache).

### Cold-expert shrinkage

At top-8 of 256, tail experts see few tokens, so raw per-expert `E[x²]` is noisy — the failure mode
where the per-expert arm loses for a reason unrelated to the hypothesis. Shrink toward the layer
statistic by token count:

```
w_e = (n_e · w_e_raw + τ · w_layer) / (n_e + τ)      # τ = pseudo-count, default 1000 tokens
```

A zero-token expert resolves exactly to `w_layer`. Each `w` is then normalized to mean 1.

`α` (applying `w^α`) is exposed but defaults to `1.0`. Unmoderated activation magnitudes span orders of
magnitude and can make k-means degenerate, with a few hot rows capturing every centroid; `α<1` is the
lever if the pure version degenerates rather than merely losing. Default stays pure so the first answer
is clean.

## Wiring

- [`encode.py`](../../src/smart_quant/encode.py) — thread `sample_weight` per expert through
  `quantize_fused_experts` into `residual_pq_quantize`. Absent weights take the existing path, so
  Phase-5/6 encodes stay byte-identical.
- [`cli.py`](../../src/smart_quant/cli.py) — `--importance-weights PATH` and
  `--importance-granularity {expert,layer}` on `encode-eval`, both off by default. Labels follow
  `wpq<bpw>-<granularity>`; the three the experiment actually runs are listed below.

## Experiment

Three sequential encodes on `pi-a100-80gb` (only one fp16 model fits at a time), as detached PPID=1
daemons, plus one forward-only calibration pass:

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

New cases in [`tests/test_codebook.py`](../../tests/test_codebook.py),
[`tests/test_encode.py`](../../tests/test_encode.py), and
[`tests/test_expert_importance.py`](../../tests/test_expert_importance.py), by area:

- **Order-1 regression safety** — `sample_weight=None` and all-ones weights are both `torch.equal`
  to today's `lloyd_kmeans` output (centroids and assignment). All-ones additionally catches
  normalization bugs that `None` would skip.
- **Weighting has the intended effect** — two separated clusters, one weighted 10×; the fitted centroid
  shifts measurably toward it.
- **`max_fit` weight alignment** — constructed so a misaligned subsample yields a detectably wrong
  centroid. This is the silent-corruption path.
- **Footprint invariance** — `pq_bpw` identical with and without weights; `quantize_fused_experts`
  realized bpw matches the unweighted encode at the same `k`.
- **Shrinkage** — zero-token expert resolves exactly to the layer statistic; high-count expert
  approaches its raw statistic.

## Plot & doc

- [`experiments/plot_quality_vs_bpw.py`](../../experiments/plot_quality_vs_bpw.py) — add
  `weighted_curve` selecting `label.startswith("wpq")`. The existing
  [`uniform_curve`](../../experiments/plot_quality_vs_bpw.py) (`endswith("-uniform")`, not `rvq`) and
  [`residual_curve`](../../experiments/plot_quality_vs_bpw.py) (`startswith("rvq")`) both miss the
  `wpq*-{expert,layer}` labels, so the new curve is additive and cannot pollute the Phase-5 line.
- [`docs/experiments/bits-per-brain.md`](../experiments/bits-per-brain.md) — Phase-7 section with
  hypothesis, matched-footprint table, and verdict.

## PR plan

One PR, `feat/weighted-pq-phase7`, off `main`. Calibration profiler + weighted quantizer + wiring +
tests + encodes + plot + doc.

## Out of scope

- **Pre-scaling (per-row `√w`) variant** — a strict superset of this design and the natural follow-up if
  weighting shows signal, but confounded as a first test.
- **Phase 6b vptq spike** — still deferred, not cancelled.
- **AQLM joint beam-search** — unchanged from Phase 6a.
- **`.pre-commit-config.yaml` ruff C901/PLR0915 gate** — repo-wide follow-up tracked separately.
