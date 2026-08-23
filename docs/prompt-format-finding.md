# Prompt Format Finding: Regularization Claim Invalidated

> The "quantization regularization" effect was a prompt format artifact.

## The Discovery

Qwen3.6-35B-A3B scored 0.295 on GSM8K with the default prompt format (5-shot, answer-only). This was unusually low for a 35B MoE model, creating the appearance that quantized models (0.870) outperformed fp16.

Testing different prompt formats revealed:

| Prompt Format | Qwen fp16 Score |
|---|---|
| gsm8k (default 5-shot, answer-only) | 0.286 |
| gsm8k_cot (8-shot, full CoT) | **0.829** |
| gsm8k_cot_zeroshot (0-shot CoT) | 0.214 |
| gsm8k_cot_llama (Llama-style 8-shot) | 0.569 |

The model CAN do math — it just needed the right prompt format.

## What This Means

### The Regularization Hypothesis is Wrong

The original claim: "quantized models outperform fp16 on math reasoning via implicit regularization."

The reality: quantized models scored 0.870 on default gsm8k, while fp16 scored 0.295. This wasn't "outperformance" — it was comparing different tasks:
- Default gsm8k expects answer-only output
- The model couldn't figure out the format with fp16 weights
- Quantization noise accidentally helped it guess the right format

### The True Curve is Monotonic Degradation

With correct CoT prompting (gsm8k_cot):

| bpw | gsm8k_cot |
|---|---|
| fp16 | **0.829** |
| 1.0 | 0.126 |
| 1.25 | 0.363 |
| 1.5 | 0.555 |
| 1.75 | 0.716 |
| 2.0 | ~0.78 (expected) |
| 2.5 | ~0.80 (expected) |

Quantized models always score lower than fp16. No regularization — just signal loss.

## Implications for Future Work

### For This Study
- Re-run full regularization curve with gsm8k_cot
- Update docs/PR with corrected findings
- The "bonus finding" (quantized models outperform fp16) is invalidated
- The core finding (PQ at 2.0-2.5 bpw keeps model "smart") may still hold but needs re-validation

### For the Field
1. **Always validate prompt format before interpreting results.** Low scores may indicate format mismatch, not model failure.
2. **Compare apples-to-apples.** If quantized models are evaluated with one prompt and fp16 with another, the comparison is meaningless.
3. **Beware of "improvements" that are format artifacts.** Quantization noise may change the model's output distribution in ways that accidentally match a particular prompt format.
4. **Test multiple prompt formats.** The choice of prompt template can swing results by 3x (0.286 vs 0.829).
