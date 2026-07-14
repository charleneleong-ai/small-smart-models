"""Evaluation: sliding-window perplexity (fast smoke) + lm-eval task accuracy."""
from __future__ import annotations

import subprocess
from pathlib import Path

import torch


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
