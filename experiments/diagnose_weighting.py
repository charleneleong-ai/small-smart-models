"""Why activation weighting failed in Phase 7, measured on real expert tensors.

Two questions the end-to-end encodes could not answer:

1. Does weighted k-means actually redirect error toward the channels importance names?
2. Is E[x^2] correlated with where the quantizer errs in the first place?

Reads fused expert weights straight from the safetensors shards — instantiating 35B of model
to inspect a handful of tensors is not worth the wait — and pairs them with the per-expert
importance produced by `smart-quant profile-activations`.

Matches the real encode: k from `centroids_for_bits`, iters=10, max_fit=max(4096, 8k).

    PYTHONPATH=src python experiments/diagnose_weighting.py --importance <path.pt>
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import typer
from huggingface_hub import snapshot_download
from safetensors import safe_open

from smart_quant.codebook import pq_dequantize, pq_quantize
from smart_quant.encode import centroids_for_bits

app = typer.Typer(add_completion=False)


def expert_slice(repo: str, idx: dict[str, str], layer: int, proj: str):
    """Open the shard holding one layer's fused projection, returning a lazy expert slicer."""
    key = sorted(k for k in idx if f".layers.{layer}.mlp.experts.{proj}" in k)[0]
    shard = snapshot_download(repo, allow_patterns=[idx[key]])
    return key, safe_open(os.path.join(shard, idx[key]), framework="pt")


def weight_map(repo: str) -> dict[str, str]:
    snap = snapshot_download(repo, allow_patterns=["*.json"])
    return json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]


@app.command()
def main(
    importance: Path = typer.Option(..., help="Per-expert .pt from profile-activations."),
    repo: str = typer.Option("Qwen/Qwen3.6-35B-A3B"),
    layers: str = typer.Option("0,13,26", help="Comma-separated decoder layer indices."),
    experts: str = typer.Option("0,128", help="Comma-separated expert indices."),
    bits: float = typer.Option(2.5),
    device: str = typer.Option("cuda"),
) -> None:
    idx = weight_map(repo)
    imp = torch.load(importance, weights_only=True)
    k = centroids_for_bits(bits, 4)
    ls = [int(x) for x in layers.split(",")]
    es = [int(x) for x in experts.split(",")]

    print(f"k={k} ({bits} bpw)   iters=10   max_fit={max(4096, k * 8)}\n")
    print(f"{'layer':>5} {'exp':>4} {'util%':>6} {'corr':>7} {'top1%':>8} {'bot50%':>8} "
          f"{'wMSE gain':>10} {'w-err top1%':>12}")
    for layer in ls:
        key, f = expert_slice(repo, idx, layer, "gate_up_proj")
        with f:
            sl = f.get_slice(key)
            for e in es:
                w = sl[e].clone().detach().float().to(device)
                cw = imp[f"{layer}.gate_up_proj"][e].float().to(device)
                codes_u, cb_u = pq_quantize(w, 4, k, iters=10, max_fit=max(4096, k * 8))
                recon_u = pq_dequantize(codes_u, cb_u)
                recon_w = pq_dequantize(*pq_quantize(
                    w, 4, k, iters=10, max_fit=max(4096, k * 8), channel_weight=cw))
                eu = (recon_u - w).pow(2).mean(0)
                ew = (recon_w - w).pow(2).mean(0)
                hi, lo = cw > cw.quantile(0.99), cw < cw.quantile(0.50)
                print(f"{layer:>5} {e:>4} {100 * codes_u.unique().numel() / k:>5.1f}% "
                      f"{torch.corrcoef(torch.stack([cw, eu]))[0, 1]:>7.3f} "
                      f"{100 * (ew[hi].mean() / eu[hi].mean() - 1):>+7.1f}% "
                      f"{100 * (ew[lo].mean() / eu[lo].mean() - 1):>+7.1f}% "
                      f"{100 * (1 - (cw * ew).mean() / (cw * eu).mean()):>9.1f}% "
                      f"{100 * (cw * eu)[hi].sum() / (cw * eu).sum():>11.1f}%")


if __name__ == "__main__":
    app()
