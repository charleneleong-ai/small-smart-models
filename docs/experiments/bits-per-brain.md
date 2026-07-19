# bits-per-brain — codebook vs. imatrix quantization on a 3B-active MoE

> How few bits per weight before the "brain" stops working — and does *how* you spend
> those bits (learned codebook vs. importance matrix) matter more than the count?

## Question

At equal memory footprint, does a **learned-codebook** 2-bit quant (VPTQ / AQLM) beat
**scalar block quant + importance matrix** (Unsloth Dynamic GGUF) on a Mixture-of-Experts
model with a tiny active path?

Target model: [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) —
35B total, **3B active** per token. Confirmed from `config.json`: `qwen3_5_moe`,
multimodal (`Qwen3_5MoeForConditionalGeneration`, vision tower + text stack), 40 decoder
layers, **256 experts, top-k 8**, `moe_intermediate_size` 512, one shared expert, hybrid
linear/full attention (Gated DeltaNet, full attention every 4th layer). Highest Artificial
Analysis Intelligence Index (32) of any 2026 open MoE that fits on a single A100-80GB for
*encoding*, and AA's designated efficiency leader — which is exactly what makes extreme
low-bit interesting: a tiny active path has the least redundancy to absorb quant error.

## Hypothesis

On **dense** models, codebook quant clearly wins at 2-bit (the published VPTQ/AQLM result).
On a **3B-active MoE** the edge should *shrink or vanish*, because importance-matrix
calibration is unusually strong here: expert usage is heavily skewed, so imatrix
automatically pours precision into the hot + shared experts. The interesting outcome is
either direction:

- **VPTQ wins** → codebook representation beats scalar even on the hardest MoE case → worth the encode cost.
- **VPTQ ties / loses** → imatrix's expert-aware allocation is what actually matters at this scale → cheaper methods suffice.

## The confound we must control

"Unsloth Dynamic imatrix" bundles **two** independent levers:

| Lever | Unsloth UD-IQ2 | Vanilla VPTQ/AQLM |
|---|---|---|
| Weight representation | scalar, per-block | learned codebook |
| Bit allocation | **dynamic** (per-layer, importance-driven) | uniform |

A naive "UD-IQ2_M vs VPTQ-2bit" comparison tests *both axes at once* and can't attribute
the result. So we run VPTQ **two ways**: uniform allocation (isolates representation) and
**expert-importance-aware** allocation (matches Unsloth's dynamic lever). Only the second is
a fair head-to-head.

## Baselines (the bar to beat)

All exist on the Hub today — download, don't recompute:

| Build | ~bpw | Size | Role |
|---|---|---|---|
| fp16 base | 16 | ~70 GB | quality ceiling |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 4 | ~19 GB | 4-bit reference |
| `unsloth/…-GGUF` `UD-IQ2_M` | ~2.6 | **11.5 GB** | **primary target** |
| `unsloth/…-GGUF` `UD-IQ2_XXS` | ~2.2 | 10.8 GB | tight-footprint target |
| `unsloth/…-GGUF` `UD-IQ1_M` | ~1.75 | 10.0 GB | stretch target |

## Method under test

1. **VPTQ** (primary — best-maintained codebook toolkit in 2026, simpler kernels than AQLM)
   on the routed expert FFNs; router, attention, shared expert, embeddings kept ≥4-bit.
2. **AQLM** (secondary, if time) — additive codebooks, more expressive, slower to encode.

Two allocations each: `uniform` and `expert-importance` (see below).

## Expert-importance-aware allocation

The novel bit. Profile expert activation frequency over the calibration set by hooking each
MoE router, then allocate bits per expert proportional to usage: hot experts get ~3-bit
codebooks, cold experts get ~1.5-bit, holding the *average* bpw fixed to match the target
footprint. This gives VPTQ the same "spend bits where they're used" advantage imatrix gets
for free. Implemented in [`smart_quant.expert_importance`](../../src/smart_quant/expert_importance.py).

## Footprint matching

Compare only at **equal file bytes ±3%** (`smart_quant.footprint.match_tolerance`). bpw is
measured against total params so the number is comparable across formats. A method that lands
0.2 bpw lighter must be re-encoded to the target budget before its quality counts.

## Evaluation

Two tiers, cheapest first so a bad quant is caught early:

- **Perplexity** — wikitext-2 + a C4 slice, sliding-window
  ([`smart_quant.eval.sliding_window_perplexity`](../../src/smart_quant/eval.py)). Fast smoke signal.
- **Task accuracy** — via `lm-eval-harness`: GSM8K, MMLU-Pro (subset), GPQA-Diamond. These
  mirror the reasoning/knowledge axes AA weights; skip the expensive agentic benches for a
  pet project.

Primary metric: **quality retention vs. fp16 at matched footprint** — i.e. Δ(perplexity) and
Δ(task acc) for each {method × allocation} at the IQ2_M byte budget, plotted against bpw.

## Success criteria

The project succeeds on a *clear answer*, not on VPTQ winning:

- Reproduce fp16 + UD-IQ2_M baselines within noise of published numbers.
- Produce VPTQ-uniform and VPTQ-expert-aware builds at the IQ2_M footprint.
- State, with CIs, whether expert-aware VPTQ beats UD-IQ2_M — and by how much.

## Results

### Phase 2 — expert-usage profiling (Qwen3.6-35B-A3B, 512 C4 rows, A100, transformers 5.14.1)

All 40 MoE layers profiled (256 experts, top-8 routing). Usage is **moderately skewed**, not
sharply concentrated:

| metric | measured | uniform baseline |
|---|---|---|
| hottest-expert share (mean / max) | 2.58% / 3.52% | 0.39% |
| top-8 experts' share (mean) | 15.3% | 3.1% |
| normalized routing entropy (mean) | 0.901 | 1.000 |

A stable hot set recurs across layers (globally hottest experts: 71, 206, 7, 95, 220, 14, 250,
67), but entropy near 0.9 means the long tail still sees real traffic (~8.4M selections over 256
experts — well-sampled, not undercounting).

**Implication:** importance-aware bit allocation has *real but modest* signal — the hot experts
warrant more bits, yet the tail isn't dead. So the uniform-vs-expert-aware gap (and imatrix's
edge over uniform codebook) may be smaller than the "MoE usage is heavily skewed" prior assumed;
a small gap would itself be a finding. Artifact: `experiments/bits-per-brain/expert_freq.pt` (box).
Caveat: single calibration domain (C4) — domain-specific calibration could shift the hot set.

## Phases

1. **Baselines** — load fp16, run eval harness; pull + eval UD-IQ2_M and AWQ-4bit. (fits 80 GB)
2. **Profile** ✅ — expert-usage histogram over calibration set. Routers identified by the MoE
   block's `gate` submodule (name-matched, so it survives both the transformers <=4
   `nn.Linear` router and the >=5 `Qwen3MoeTopKRouter` refactor, plus the multimodal
   nesting) — see [`smart_quant.expert_importance`](../../src/smart_quant/expert_importance.py).
   Verified on the A100 against a real transformers-5 Qwen3-MoE module tree.
3. **Encode** — VPTQ uniform, then expert-aware, at the IQ2_M budget.
4. **Measure** — same harness; footprint-matched comparison table + quality-vs-bpw plot.
5. **Write up** — `docs/experiments/vptq-vs-imatrix.md`, W&B run group `ssm-vptq`.

## Hardware & budget

Single A100-80GB (`pi-a100-80gb`). fp16 load ~70 GB leaves headroom for calibration.
Encode passes are the cost: VPTQ codebook fitting over the experts is GPU-hours, not
minutes. Rough estimate 1–2 GPU-days total across both allocations + AQLM stretch.

## Risks

- **VPTQ tooling may not support `qwen3_5_moe`** out of the box — verify layer plumbing in
  phase 2 before committing to the encode. Fallback: AQLM, or hand-roll the codebook fit on
  expert `nn.Linear`s.
- **Encode never converges at the tiny-active budget** — report it; a negative result on
  feasibility is still a finding.
- **imatrix already captures most of the win** — the expected null result; make sure the
  eval has enough power (CIs) to call a tie a tie.
