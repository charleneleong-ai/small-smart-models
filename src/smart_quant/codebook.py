"""Product-quantization codebook quantizer for weight matrices.

Arch-agnostic — operates on any 2D weight, so it sidesteps the per-architecture support
that GGUF / GPTQ / VPTQ tooling needs (which `qwen3_5_moe` currently lacks). Each weight row
is split into `sub_dim`-wide sub-vectors and each is stored as an index into a k-means
codebook.

Codebook sharing is the knob that reaches low bpw: with one codebook per group the fp16
codebook storage rivals the index storage (~4 bpw on 2048x512 experts); sharing a single
codebook across all groups amortizes it to ~nominal log2(n_centroids)/sub_dim (~2 bpw).
"""
from __future__ import annotations

import math

import torch

__all__ = ["lloyd_kmeans", "pq_quantize", "pq_dequantize", "pq_bpw", "residual_pq_quantize"]


def lloyd_kmeans(x: torch.Tensor, k: int, iters: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's k-means over rows of x (n, d), deterministic linspace init. Returns
    (centroids (k, d), assignment (n,)). Empty clusters keep their previous centroid.
    Centroid update is vectorized (index_add) — no Python loop over k."""
    n = x.shape[0]
    centroids = x[torch.linspace(0, n - 1, k).round().long()].clone()
    ones = torch.ones(n, device=x.device, dtype=x.dtype)
    for _ in range(iters):
        idx = torch.cdist(x, centroids).argmin(dim=1)
        sums = torch.zeros_like(centroids).index_add_(0, idx, x)
        counts = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(0, idx, ones)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    idx = torch.cdist(x, centroids).argmin(dim=1)
    return centroids, idx


def pq_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    n_centroids: int,
    iters: int = 10,
    share_codebook: bool = True,
    max_fit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Product-quantize a (out, in) weight. Returns (codes (out, groups) long, codebooks).
    With share_codebook, one codebook is fit over all sub-vectors -> codebooks is
    (n_centroids, sub_dim); otherwise one per group -> (groups, n_centroids, sub_dim).
    max_fit caps how many sub-vectors the shared codebook is *fit* on (a strided subsample);
    all sub-vectors are still assigned. This keeps k-means tractable at scale — fit cost
    drops from ~500k points/expert to max_fit, while assignment is a single pass."""
    out, in_ = weight.shape
    if in_ % sub_dim:
        raise ValueError(f"in_features {in_} not divisible by sub_dim {sub_dim}")
    groups = in_ // sub_dim
    subvecs = weight.reshape(out, groups, sub_dim)

    if share_codebook:
        pool = subvecs.reshape(-1, sub_dim).float()
        fit = pool
        if max_fit is not None and pool.shape[0] > max_fit:
            fit = pool[torch.linspace(0, pool.shape[0] - 1, max_fit).round().long()]
        centroids = lloyd_kmeans(fit, n_centroids, iters)[0]
        idx = torch.cdist(pool, centroids).argmin(dim=1)
        return idx.reshape(out, groups), centroids.to(weight.dtype)

    codes = torch.empty(out, groups, dtype=torch.long)
    codebooks = torch.empty(groups, n_centroids, sub_dim, dtype=weight.dtype)
    for g in range(groups):
        centroids, idx = lloyd_kmeans(subvecs[:, g, :].float(), n_centroids, iters)
        codebooks[g] = centroids.to(weight.dtype)
        codes[:, g] = idx
    return codes, codebooks


def pq_dequantize(codes: torch.Tensor | list[torch.Tensor],
                  codebooks: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct the (out, in) weight. A single (codes, codebooks) pair reconstructs one
    stage (shared 2D or per-group 3D codebook); a (codes_list, codebooks_list) pair sums the
    per-stage reconstructions of a residual quantization."""
    if isinstance(codes, list):
        return sum(pq_dequantize(c, cb) for c, cb in zip(codes, codebooks))
    out, groups = codes.shape
    if codebooks.dim() == 2:  # shared: one codebook indexes every group
        recon = codebooks[codes]
    else:  # per-group
        recon = torch.stack([codebooks[g][codes[:, g]] for g in range(groups)], dim=1)
    return recon.reshape(out, -1)


def residual_pq_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    stage_centroids: list[int],
    iters: int = 10,
    share_codebook: bool = True,
    max_fit: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Multi-stage residual product quantization: stage 0 quantizes `weight`, each later
    stage quantizes the running residual `weight - sum(recon so far)`. Returns per-stage
    (codes_list, codebooks_list); `stage_centroids=[k]` reproduces a single `pq_quantize`."""
    codes_list: list[torch.Tensor] = []
    codebooks_list: list[torch.Tensor] = []
    residual = weight
    last = len(stage_centroids) - 1
    for stage, k in enumerate(stage_centroids):
        codes, codebook = pq_quantize(residual, sub_dim, k, iters, share_codebook, max_fit)
        codes_list.append(codes)
        codebooks_list.append(codebook)
        if stage < last:  # final stage's residual is never read — skip the dequantize
            residual = residual - pq_dequantize(codes, codebook).to(weight.dtype)
    return codes_list, codebooks_list


def pq_bpw(
    out: int, in_: int, sub_dim: int, n_centroids: int | list[int], share_codebook: bool = True
) -> float:
    """Effective bits-per-weight including fp16 codebook storage, summed over residual stages.
    `n_centroids` may be a single int (one stage) or a per-stage list. Sharing a single codebook
    (vs one per group) is what drops the overhead from ~index-storage to negligible."""
    stages = n_centroids if isinstance(n_centroids, list) else [n_centroids]
    groups = in_ // sub_dim
    n_codebooks = 1 if share_codebook else groups
    total_bits = sum(
        out * groups * math.log2(k) + n_codebooks * k * sub_dim * 16 for k in stages
    )
    return total_bits / (out * in_)
