"""Evaluation: sliding-window perplexity (fast smoke) + lm-eval task accuracy."""
from __future__ import annotations

import subprocess
from pathlib import Path

import torch


def load_causal_lm(model_id: str, **kwargs):
    """Load a text-generation model, tolerating multimodal ConditionalGeneration archs
    (e.g. Qwen3.5-MoE) that `AutoModelForCausalLM` does not map to a CausalLM class."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)


def sliding_window_perplexity(
    model,
    tokenizer,
    text: str,
    max_length: int = 4096,
    stride: int = 2048,
    device: str = "cuda",
) -> float:
    """Standard HF sliding-window perplexity — only the last `stride` tokens of each
    window contribute, so overlapping context is not double-counted."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        window = input_ids[:, begin:end]
        targets = window.clone()
        targets[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(window, labels=targets).loss
        nlls.append(loss * trg_len)
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / end).item()


def run_lm_eval(
    model_args: str,
    tasks: list[str],
    out_dir: Path,
    limit: int | None = None,
) -> Path:
    """Shell out to lm-evaluation-harness; return the results directory.

    model_args is the harness --model_args string, e.g.
    'pretrained=Qwen/Qwen3.6-35B-A3B,dtype=bfloat16'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--output_path", str(out_dir),
        "--batch_size", "auto",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
    return out_dir


# Capability battery: reasoning, two commonsense axes, math, broad knowledge. All leaf eval on
# a 35B (or grouped, `mmlu` aggregates to one number), small enough to afford on every build row.
CAPABILITY_BATTERY = ("arc_challenge", "hellaswag", "winogrande", "gsm8k", "mmlu")
_METRIC_PRIORITY = ("acc_norm", "acc", "exact_match", "strict_match", "pass@1")


def primary_accuracy(metrics: dict[str, float]) -> float:
    """Headline accuracy for a task from an lm-eval per-task metric dict.

    Multiple-choice tasks report `acc` / `acc_norm` (HellaSwag-style continuations list both),
    generation tasks `exact_match` / `strict_match`, code tasks `pass@1`. lm-eval 0.4 keys the
    dict `metric,aggregation`, so the bare metric is the part before the comma; `_stderr` keys
    are skipped by exact matching on that bare name. The priority picks the length-normalized
    multiple-choice number where both exist, and is what `mmlu`'s group aggregate resolves
    through too."""
    for metric in _METRIC_PRIORITY:
        key = next((k for k in metrics if k.split(",")[0] == metric), None)
        if key is not None:
            return round(float(metrics[key]), 4)
    raise KeyError(f"no known accuracy metric in {sorted(metrics)}")


def run_task_battery(
    model,
    tokenizer,
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, float]:
    """lm-eval task accuracy against an already-loaded (e.g. fake-quantized) model, in memory.

    The model never round-trips to disk — that is what lets a 2-bit build be scored on the same
    artifact its perplexity row came from. Returns `{task: headline accuracy}`; a grouped task
    like `mmlu` resolves to its aggregated entry in `results["results"]`. The lm_eval import is
    lazy so this module stays importable in CI, which installs only torch."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    requested = set(tasks or CAPABILITY_BATTERY)
    results = simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer),
        tasks=sorted(requested),
        limit=limit,
        confirm_run_unsafe_code=True,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results — task load failed")
    return {name: primary_accuracy(metrics)
            for name, metrics in results["results"].items() if name in requested}
