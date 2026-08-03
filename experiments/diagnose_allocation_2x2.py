"""Why does importance allocation help GGUF but hurt our shared-codebook PQ?

The field allocates more bits to hot experts (BitsMoE, MC-MoE, imatrix) and reports gains; Phase
3 measured the same water-fill *hurting* a shared-codebook PQ, monotonically in strength. Two
families differ in where their error comes from:

  scalar (per-row)   closed-form scales, no global fit — error tracks the rate (~6 dB/bit)
  PQ (d=4 learned)   Lloyd fit on a finite sample — error saturates below what the rate promises

This 2×2 isolates that axis. If PQ is *fit-limited* while scalar is *rate-limited*, the identical
usage-allocated water-fill should HELP scalar and HURT PQ — the interaction the phases have only
seen from the PQ side. Both cells matched on mean allocated bpw; error on a disjoint eval half
(odd output rows) of real Qwen3.6-35B-A3B layer-13 `gate_up_proj`; the allocation signal is the
real Phase-2 routing frequency (`experiments/bits-per-brain/expert_freq.pt`).
"""
from __future__ import annotations

import json
import math
import os

import torch
import typer
from huggingface_hub import snapshot_download
from safetensors import safe_open

from smart_quant.codebook import assign, lloyd_kmeans
from smart_quant.expert_importance import bits_from_frequency
from smart_quant.lattice import strided_indices
from smart_quant.scalar import scalar_quantize

REPO, SUB_DIM, ITERS = "Qwen/Qwen3.6-35B-A3B", 4, 10
ASSIGN_CHUNK = 4096
RATES = (1.75, 2.0, 2.25, 2.5, 2.75, 3.0)


def rel_err(recon: torch.Tensor, ref: torch.Tensor) -> float:
    return float((recon - ref).norm() / ref.norm())


def assign_chunked(pool: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(pool)
    for i in range(0, pool.shape[0], ASSIGN_CHUNK):
        blk = pool[i:i + ASSIGN_CHUNK]
        out[i:i + ASSIGN_CHUNK] = book[assign(blk, book)]
    return out


def pq_rel_err(fit: torch.Tensor, eval_: torch.Tensor, bits: float) -> float:
    """Fit a shared codebook on `fit` (sampled to the shipped max_fit budget), assign and score
    `eval_` — disjoint halves, so the high-k cells can't win by memorising the eval set."""
    k = int(2 ** (bits * SUB_DIM))
    max_fit = max(4096, k * 8)
    sample = fit[strided_indices(fit.shape[0], max_fit, fit.device)]
    book = lloyd_kmeans(sample, k, ITERS)[0]
    return rel_err(assign_chunked(eval_, book), eval_)


def scalar_rel_err(weight: torch.Tensor, bits: float, eval_: torch.Tensor) -> float:
    """Per-row scalar on the full tensor (scales are closed-form, nothing learned to overfit),
    scored on the same odd-row eval half as PQ."""
    rec = scalar_quantize(weight, bits)[1::2].reshape(-1, SUB_DIM)
    return rel_err(rec, eval_)


def load_weights(layer: int, n_experts: int) -> torch.Tensor:
    snap = snapshot_download(REPO, allow_patterns=["*.json"])
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    key = sorted(x for x in idx if f".layers.{layer}.mlp.experts.gate_up_proj" in x)[0]
    shard = snapshot_download(REPO, allow_patterns=[idx[key]])
    with safe_open(os.path.join(shard, idx[key]), framework="pt") as f:
        return f.get_slice(key)[:n_experts].clone().detach().float().cuda()


def load_freq(path: str, layer: int, n_experts: int) -> torch.Tensor:
    freq = torch.load(path, weights_only=True)
    keys = [k for k in freq if f"layers.{layer}." in k]
    if not keys:
        raise FileNotFoundError(f"no layer-{layer} router frequency in {path}")
    return freq[keys[0]][:n_experts]


def mean_bpw(b: torch.Tensor) -> float:
    """Realized per-element rate: scalar realizes round(2^bits) levels, PQ the index rate."""
    scalar = torch.tensor([math.log2(max(2, round(2 ** float(x)))) for x in b]).mean().item()
    return scalar


def main(
    freq: str = typer.Option("experiments/bits-per-brain/expert_freq.pt"),
    avg_bits: float = typer.Option(2.0),
    lo: float = typer.Option(1.5),
    hi: float = typer.Option(3.0),
    layer: int = typer.Option(13),
    n_experts: int = typer.Option(32),
) -> None:
    w = load_weights(layer, n_experts)
    freqs = load_freq(freq, layer, n_experts)
    fits = [w[e][0::2].reshape(-1, SUB_DIM) for e in range(n_experts)]
    evals = [w[e][1::2].reshape(-1, SUB_DIM) for e in range(n_experts)]
    alloc = bits_from_frequency(freqs, avg_bits, lo=lo, hi=hi)
    uniform = torch.full((n_experts,), float(avg_bits))

    print(f"layer {layer} · {n_experts} experts · d={SUB_DIM} · avg {avg_bits} bpw · "
          f"span [{lo}, {hi}] · eval = odd rows, disjoint from PQ fit\n", flush=True)

    def scalar_cell(b: torch.Tensor) -> float:
        return sum(scalar_rel_err(w[e], float(b[e]), evals[e]) for e in range(n_experts))

    def pq_cell(b: torch.Tensor) -> float:
        return sum(pq_rel_err(fits[e], evals[e], float(b[e])) for e in range(n_experts))

    rows = {
        "scalar": (scalar_cell(uniform), scalar_cell(alloc)),
        "pq": (pq_cell(uniform), pq_cell(alloc)),
    }
    print(f"{'family':<14} {'uniform':>10} {'usage-alloc':>12} {'Δ':>9}")
    for family, (u, a) in rows.items():
        delta = (a - u) / u * 100
        print(f"{family:<14} {u:>10.4f} {a:>12.4f} {delta:>+8.1f}%", flush=True)

    print(f"\nrealized mean bpw (scalar level rate): uniform {mean_bpw(uniform):.2f} / "
          f"alloc {mean_bpw(alloc):.2f} · pq index rate: uniform {avg_bits:.2f} / "
          f"alloc {alloc.mean():.2f}")

    print(f"\nR-D curve at uniform rate (rel L2, sum over {n_experts} experts):")
    print(f"{'bpw':>6} {'scalar':>10} {'pq':>10} {'pq slope dB/bit':>16}")
    prev: float | None = None
    for b in RATES:
        s = scalar_cell(torch.full((n_experts,), b))
        p = pq_cell(torch.full((n_experts,), b))
        slope = (20 * math.log10(p / prev) / (b - prev)) if prev else float("nan")
        print(f"{b:>6.2f} {s:>10.4f} {p:>10.4f} {slope:>16.1f}", flush=True)
        prev = b


if __name__ == "__main__":
    typer.run(main)
