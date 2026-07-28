# Phase 6 — residual vector quantization (second-order codebook)

**Study:** bits-per-brain · **Branch:** `feat/residual-vq-phase6` (stacked on `feat/equal-footprint-bpw`, PR #6)
**Date:** 2026-07-28 · **Status:** design approved, pending spec review

## Problem

Phase 5 established that at matched ~2.6 bpw our first-order product-quantization (PQ) beats the
Unsloth imatrix UD-IQ2_M baseline (interp. 6.19 vs normalized 6.38 wikitext-2 ppl). The open Phase-6
question: does a **second-order codebook** — the VPTQ/AQLM insight, where a second stage quantizes the
*residual* of the first — widen that margin at the **same** footprint, or does splitting a fixed bit
budget across two coarse codebooks cost more than one fine codebook buys?

## The reframe (why "residual-VQ", not the vptq library)

The original Phase-6 line item was "VPTQ/AQLM". Investigation found the real `vptq` library is a dead
end here for the same reason GGUF/GPTQ were (Phase 3 pivot): its quantizer is an architecture-specific
branch with no `qwen3_5_moe` support. So Phase 6 takes the *idea* (learned residual codebook) and builds
it into our arch-agnostic PQ, which operates on any 2D weight. The vptq-library feasibility question is
split off as **Phase 6b** — a separate, time-boxed spike in its own PR. This spec is Phase 6a only.

## Design

Three surgical changes to [`src/smart_quant/codebook.py`](../../src/smart_quant/codebook.py),
generalizing the existing single-stage path to M stages. The M=1 path stays byte-identical.

### 1. `residual_pq_quantize` — new

```python
def residual_pq_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    stage_centroids: list[int],
    iters: int = 10,
    share_codebook: bool = True,
    max_fit: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
```

Sequential residual fit: stage 0 quantizes `weight`; each later stage quantizes the running residual
`weight - Σ recon_so_far` by calling the existing `pq_quantize`. Returns `(codes_list, codebooks_list)`,
one entry per stage. `stage_centroids=[k]` (length 1) is exactly today's single-stage call. Sequential
(greedy) residual, **not** AQLM joint beam-search — YAGNI; the joint variant is a later stretch if the
sequential margin is promising.

### 2. `pq_dequantize` — generalized to accept stage lists

Overloaded on input type: a single `(codes, codebooks)` pair reconstructs as today; a
`(codes_list, codebooks_list)` pair sums the per-stage reconstructions `Σ_k codebooks[k][codes[k]]`. The
single-pair path is untouched so all existing callers and the encode fast-path keep working.

### 3. `pq_bpw` — generalized to M stages

```python
def pq_bpw(out, in_, sub_dim, stage_centroids, share_codebook=True) -> float:
```

`stage_centroids` becomes a list; the realized bpw is the sum over stages of
`(index_bits_k + codebook_bits_k) / (out*in_)`. An `int` is still accepted (coerced to `[k]`) so the
Phase-5 signature keeps working. This is the honest accounting knob that keeps the matched-footprint
comparison from cheating — the two stages' realized bpw is what we hold fixed, not a nominal target.

### Matched-budget discipline

The comparison holds `sub_dim` fixed and **splits the same index-bit budget across the two stages**
rather than letting the residual stage add bits on top — a single `k=1024` codebook (10 index-bits/group)
vs two `k=32` codebooks (5+5 index-bits/group). Under a shared codebook the codebook-storage term is
small and differs slightly between the two (two coarse books store less than one fine book), so the k
values are **tuned so `pq_bpw` reports matching total realized bpw** before either result counts — the
index split is the intuition, `pq_bpw` is the referee.

## Wiring

- [`src/smart_quant/encode.py`](../../src/smart_quant/encode.py) — thread `codebook_order: int = 1`
  through `quantize_fused_experts`; when `order > 1` it calls `residual_pq_quantize` with a
  budget-split `stage_centroids` derived from the per-expert bit target, else the existing single-stage
  call. `order=1` is the default → no behavior change for Phase-5 encodes.
- [`src/smart_quant/cli.py`](../../src/smart_quant/cli.py) — add `--codebook-order` (default 1) to
  `encode-eval`; new rows get labels like `rvq26-uniform` so the plot can pick them out.

## Experiment

Two matched-footprint residual encodes on `pi-a100-80gb` (sequential — only one fp16 model fits at a
time), as detached PPID=1 daemons:

- `rvq20-uniform` — ~2.0 realized expert_bpw, order=2
- `rvq26-uniform` — ~2.6 realized expert_bpw, order=2 (head-to-head with the imatrix locus)

Two points so the plot shows a **second curve**, not a single dot. Rows append to
`experiments/bits-per-brain/results.jsonl` (gitignored, regenerated). Success = the residual curve sits
below the first-order curve at matched bpw; a null result (residual ≥ first-order) is still a publishable
Phase-6 verdict.

## Testing

New cases in `tests/test_codebook.py` (one file per area, classes per sub-feature):

- **Residual reconstruction** — per-stage residual error strictly decreases (stage 2 recon closer than
  stage 1); M=1 residual output is byte-identical to `pq_quantize`/`pq_dequantize`.
- **M-stage `pq_bpw`** — two `k` stages sum to the expected realized bpw; `int` and `[int]` inputs agree
  (Phase-5 regression).
- **Round-trip** — `residual_pq_quantize` → generalized `pq_dequantize` shape + dtype match input weight.

## Plot & doc

- [`experiments/plot_quality_vs_bpw.py`](../../experiments/plot_quality_vs_bpw.py) — overlay the
  residual curve (rows whose label starts `rvq`) alongside the first-order uniform curve and the imatrix
  locus.
- [`docs/experiments/bits-per-brain.md`](../experiments/bits-per-brain.md) — add a Phase-6
  section (hypothesis, matched-budget table, verdict).

## PR plan

- **Phase 6a (this spec):** one PR `feat/residual-vq-phase6`, stacked on `feat/equal-footprint-bpw`
  (base retargets to `main` after PR #6 merges). Residual-VQ primitives + wiring + tests + encodes +
  plot + doc.
- **Phase 6b (deferred, separate PR):** time-boxed spike on whether `vptq>=0.0.5` can plumb
  `qwen3_5_moe` experts today. Not in this spec.

## Out of scope

- AQLM joint (beam-search) residual — only if the sequential margin justifies it.
- `.pre-commit-config.yaml` ruff C901/PLR0915 gate — repo-wide follow-up tracked separately.
