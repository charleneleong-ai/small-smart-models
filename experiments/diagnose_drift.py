"""Why GPTQ-style compensation harms a shared-codebook quantizer.

Phase 8 measured that error compensation loses to uniform PQ. This isolates the mechanism.

Compensation displaces not-yet-quantized columns, but the codebook was fit *before* it ran — so
the hypothesis is that sub-vectors drift away from their centroids as the pass proceeds. That is
falsifiable: assignment distance should grow with group index under compensation and stay flat
without it. Reads weights straight from the safetensors shards and pairs them with the per-layer
Hessian from `smart-quant profile-hessian`.

    PYTHONPATH=src python experiments/diagnose_drift.py --hessian experiments/expert_hessian.pt
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import typer
from huggingface_hub import snapshot_download
from safetensors import safe_open

from smart_quant.codebook import assign, pq_quantize
from smart_quant.compensate import damped_inverse
from smart_quant.encode import centroids_for_bits

app = typer.Typer(add_completion=False)


def octile_profile(weight: torch.Tensor, codebook: torch.Tensor, hinv_chol: torch.Tensor,
                   sub_dim: int, compensate: bool) -> torch.Tensor:
    """Mean squared distance from each sub-vector to its assigned centroid, per group, as one
    pass proceeds. The codebook is held fixed — that is the point: it was fit before the pass."""
    work, prof = weight.clone(), []
    in_ = weight.shape[1]
    for g in range(in_ // sub_dim):
        lo, hi = g * sub_dim, (g + 1) * sub_dim
        block = work[:, lo:hi]
        recon = codebook[assign(block, codebook)]
        prof.append(float((block - recon).pow(2).mean()))
        if compensate and hi < in_:
            delta = (block - recon) @ torch.linalg.inv(hinv_chol[lo:hi, lo:hi])
            work[:, hi:] -= delta @ hinv_chol[lo:hi, hi:]
    return torch.tensor(prof)


@app.command()
def main(
    hessian: Path = typer.Option(..., help="Per-layer Hessian .pt from profile-hessian."),
    repo: str = typer.Option("Qwen/Qwen3.6-35B-A3B"),
    pairs: str = typer.Option("13:18,13:237,26:109,26:218", help="layer:expert, comma-separated."),
    bits: float = typer.Option(2.5),
    sub_dim: int = typer.Option(4),
    device: str = typer.Option("cuda"),
) -> None:
    snap = snapshot_download(repo, allow_patterns=["*.json"])
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    hessians = torch.load(hessian, weights_only=True)
    k = centroids_for_bits(bits, sub_dim)

    print(f"k={k} ({bits} bpw)  sub_dim={sub_dim}\n")
    print(f"{'layer':>5} {'exp':>4} {'plain slope':>12} {'comp slope':>11} "
          f"{'comp/plain':>11} {'monotonic':>10}")
    for pair in pairs.split(","):
        layer, expert = (int(x) for x in pair.split(":"))
        key = sorted(x for x in idx if f".layers.{layer}.mlp.experts.gate_up_proj" in x)[0]
        shard = snapshot_download(repo, allow_patterns=[idx[key]])
        with safe_open(os.path.join(shard, idx[key]), framework="pt") as f:
            w = f.get_slice(key)[expert].clone().detach().float().to(device)
        u = damped_inverse(hessians[layer].to(device)).float()
        groups = w.shape[1] // sub_dim
        codebook = pq_quantize(w, sub_dim, k, iters=10, max_fit=max(4096, k * 8))[1].float()

        plain = octile_profile(w, codebook, u, sub_dim, False)
        comp = octile_profile(w, codebook, u, sub_dim, True)
        n = groups // 8
        octiles = [float(comp[i * n:(i + 1) * n].mean()) for i in range(8)]
        monotonic = all(octiles[i] <= octiles[i + 1] * 1.001 for i in range(7))
        print(f"{layer:>5} {expert:>4} {plain[-n:].mean() / plain[:n].mean():>11.2f}x "
              f"{comp[-n:].mean() / comp[:n].mean():>10.2f}x "
              f"{comp.mean() / plain.mean():>10.3f}x {str(monotonic):>10}")


if __name__ == "__main__":
    app()
