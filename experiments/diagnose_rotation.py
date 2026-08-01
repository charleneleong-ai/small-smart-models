"""Does incoherence processing help a *learned* codebook?

Every working 2-bit method rotates the weight matrix first (QuIP random Hadamard, QuaRot,
SpinQuant). This study never has, which looked like its biggest gap. This measures it before
building a phase around it.

Quantize `W` directly, versus quantize `W @ Q` under a random-sign Hadamard `Q` and map back with
`Q.T`. `Q` is orthogonal, so the layer's function is unchanged (`x -> Q.T x` absorbs it) and the
comparison is exactly matched in footprint — same k, same sub_dim, same shared codebook. Averaged
over several random sign draws, since one draw is a sample and not a result.

Reports the QuIP incoherence parameter, reconstruction MSE, and the layerwise objective under the
measured input covariance.

    PYTHONPATH=src python experiments/diagnose_rotation.py --hessian experiments/expert_hessian.pt
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import typer
from huggingface_hub import snapshot_download
from safetensors import safe_open

from smart_quant.codebook import pq_dequantize, pq_quantize
from smart_quant.encode import centroids_for_bits

app = typer.Typer(add_completion=False)


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    """Sylvester construction, normalized to orthonormal. `n` must be a power of two — expert
    tensors are 2048 and 512, so no padding is needed."""
    if n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


def incoherence(w: torch.Tensor) -> float:
    """QuIP's mu. 1.0 means energy spread perfectly evenly; larger means concentrated."""
    return float(w.abs().max() * (w.shape[0] * w.shape[1]) ** 0.5 / w.norm())


def layer_err(recon: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> float:
    """tr(D H D^T)/numel — error weighted by what the inputs actually are."""
    d = (recon - w).double()
    return float((d @ h.double() * d).sum() / d.numel())


@app.command()
def main(
    hessian: Path = typer.Option(..., help="Per-layer Hessian .pt from profile-hessian."),
    repo: str = typer.Option("Qwen/Qwen3.6-35B-A3B"),
    pairs: str = typer.Option("13:18,26:109", help="layer:expert, comma-separated."),
    bits: float = typer.Option(2.5),
    sub_dim: int = typer.Option(4),
    trials: int = typer.Option(3, help="Random sign draws per tensor."),
    device: str = typer.Option("cuda"),
) -> None:
    snap = snapshot_download(repo, allow_patterns=["*.json"])
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    hessians = torch.load(hessian, weights_only=True)
    k = centroids_for_bits(bits, sub_dim)
    max_fit = max(4096, k * 8)

    print(f"k={k} ({bits} bpw, sub_dim={sub_dim})   Q = random-sign Hadamard\n")
    print(f"{'layer':>5} {'exp':>4} {'variant':>10} {'mu':>7} {'recon MSE':>12} "
          f"{'layerwise':>12} {'vs plain':>9}")
    for pair in pairs.split(","):
        layer, expert = (int(x) for x in pair.split(":"))
        key = sorted(x for x in idx if f".layers.{layer}.mlp.experts.gate_up_proj" in x)[0]
        shard = snapshot_download(repo, allow_patterns=[idx[key]])
        with safe_open(os.path.join(shard, idx[key]), framework="pt") as f:
            w = f.get_slice(key)[expert].clone().detach().float().to(device)
        h = hessians[layer].to(device)

        recon = pq_dequantize(*pq_quantize(w, sub_dim, k, iters=10, max_fit=max_fit))
        base_mse = float((recon - w).pow(2).mean())
        print(f"{layer:>5} {expert:>4} {'plain':>10} {incoherence(w):>7.2f} {base_mse:>12.4e} "
              f"{layer_err(recon, w, h):>12.4e} {'—':>9}")

        for trial in range(trials):
            gen = torch.Generator(device=w.device).manual_seed(trial)
            signs = (torch.randint(0, 2, (w.shape[1],), generator=gen,
                                   device=w.device) * 2 - 1).float()
            q = hadamard(w.shape[1], w.device) * signs.unsqueeze(0)
            rot = pq_dequantize(*pq_quantize(w @ q, sub_dim, k, iters=10, max_fit=max_fit))
            back = rot @ q.T
            mse = float((back - w).pow(2).mean())
            print(f"{'':>5} {'':>4} {'rot ' + str(trial):>10} {incoherence(w @ q):>7.2f} "
                  f"{mse:>12.4e} {layer_err(back, w, h):>12.4e} "
                  f"{100 * (1 - mse / base_mse):>8.1f}%")


if __name__ == "__main__":
    app()
