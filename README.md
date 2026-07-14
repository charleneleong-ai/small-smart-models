# small-smart-models

How small can a model get while staying *smart*? A benchmark program exploring the
techniques that shrink intelligence-dense open models — each a self-contained study over
the smartest open MoEs of 2026, sharing one eval + footprint toolkit (`smart_quant`).

## Studies

- **[`bits-per-brain`](docs/experiments/bits-per-brain.md)** *(active)* — quantization: does
  learned-codebook 2-bit quant (VPTQ / AQLM) beat scalar importance-matrix quant (Unsloth
  Dynamic GGUF) at equal footprint on a 3B-active MoE?
- _planned_ — distillation, structured pruning, speculative decoding, expert dropping.

Target: [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — 35B total /
3B active, top Artificial Analysis Intelligence Index among single-GPU-encodable 2026 MoEs.

## Setup

```bash
uv sync                      # core deps (torch, transformers, datasets, lm-eval)
uv sync --extra quant        # heavy codebook toolkits (vptq, aqlm) — GPU box only
```

## Usage

```bash
# footprint accounting — what bpw is a given quant file?
uv run smart-quant footprint --params 35e9 --bytes 11522702304

# perplexity smoke test on a build
uv run smart-quant ppl --model unsloth/Qwen3.6-35B-A3B-GGUF --file UD-IQ2_M

# profile expert-activation frequency (drives importance-aware allocation)
uv run smart-quant profile-experts --model Qwen/Qwen3.6-35B-A3B --calib-rows 512
```

## Layout

```
docs/experiments/<study>.md      one plan per study (start: bits-per-brain.md)
src/smart_quant/                 shared toolkit: footprint · eval · expert profiling · cli
configs/<study>.yaml             per-study run config (Hydra)
experiments/<study>/             results.jsonl, plots (gitignored)
tests/                           footprint math
```
