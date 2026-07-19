"""Product-quantization codebook quantizer for weight matrices.

Arch-agnostic — operates on any 2D weight, so it sidesteps the per-architecture support
that GGUF / GPTQ / VPTQ tooling needs (which `qwen3_5_moe` currently lacks). Each weight row
is split into `sub_dim`-wide sub-vectors; per group, k-means learns `n_centroids` codebook
entries and each sub-vector is stored as an index. Nominal bits-per-weight from the indices
is log2(n_centroids)/sub_dim; the fp16 codebooks add overhead that `pq_bpw` accounts for.
"""
from __future__ import annotations

import math

import torch

__all__ = ["lloyd_kmeans", "pq_quantize", "pq_dequantize", "pq_bpw"]


def lloyd_kmeans(x: torch.Tensor, k: int, iters: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's k-means over rows of x (n, d), deterministic linspace init. Returns
    (centroids (k, d), assignment (n,)). Empty clusters keep their previous centroid."""
    n = x.shape[0]
    centroids = x[torch.linspace(0, n - 1, k).round().long()].clone()
    for _ in range(iters):
        idx = torch.cdist(x, centroids).argmin(dim=1)
        for j in range(k):
            mask = idx == j
            if mask.any():
                centroids[j] = x[mask].mean(dim=0)
    idx = torch.cdist(x, centroids).argmin(dim=1)
    return centroids, idx


def pq_quantize(
    weight: torch.Tensor, sub_dim: int, n_centroids: int, iters: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Product-quantize a (out, in) weight. Returns (codes (out, groups) long,
    codebooks (groups, n_centroids, sub_dim))."""
    out, in_ = weight.shape
    if in_ % sub_dim:
        raise ValueError(f"in_features {in_} not divisible by sub_dim {sub_dim}")
    groups = in_ // sub_dim
    w = weight.reshape(out, groups, sub_dim)
    codes = torch.empty(out, groups, dtype=torch.long)
    codebooks = torch.empty(groups, n_centroids, sub_dim, dtype=weight.dtype)
    for g in range(groups):
        centroids, idx = lloyd_kmeans(w[:, g, :].float(), n_centroids, iters)
        codebooks[g] = centroids.to(weight.dtype)
        codes[:, g] = idx
    return codes, codebooks


def pq_dequantize(codes: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Reconstruct the (out, in) weight from PQ codes + codebooks."""
    _, groups = codes.shape
    recon = torch.stack([codebooks[g][codes[:, g]] for g in range(groups)], dim=1)
    return recon.reshape(codes.shape[0], -1)


def pq_bpw(out: int, in_: int, sub_dim: int, n_centroids: int) -> float:
    """Effective bits-per-weight including fp16 codebook storage. Codebook overhead is
    amortized across `out` rows, so it shrinks as the weight matrix grows."""
    groups = in_ // sub_dim
    index_bits = out * groups * math.log2(n_centroids)
    codebook_bits = groups * n_centroids * sub_dim * 16
    return (index_bits + codebook_bits) / (out * in_)
