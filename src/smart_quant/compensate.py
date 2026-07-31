from __future__ import annotations

import torch

from smart_quant.codebook import assign, pq_quantize, pq_dequantize

__all__ = ["damped_inverse", "compensated_quantize", "compensated_quantize_fused"]


def damped_inverse(h: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor of the damped `H^-1`, in float64.

    `damp * mean(diag(H))` on the diagonal is GPTQ's standard preconditioning. It is
    load-bearing here, not decorative: layer 0's covariance has condition ~1e5 and any
    per-expert estimate over few routed tokens is outright rank-deficient. Dead input channels
    (all-zero column, which real experts have) get a unit diagonal so they contribute no
    compensation rather than producing a non-positive-definite matrix."""
    n = h.shape[0]
    h = h.double().clone()
    d = torch.arange(n, device=h.device)
    dead = torch.diagonal(h) == 0
    h[dead, dead] = 1.0
    h[d, d] += damp * torch.diagonal(h).mean()
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(hinv, upper=True)


def compensated_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    n_centroids: int,
    hinv_chol: torch.Tensor,
    iters: int = 10,
    max_fit: int | None = None,
    rounds: int = 3,
    compensate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Quantize a (out, in) weight group-by-group, pushing each group's error onto the columns
    not yet quantized. Returns (codes, codebook, per-round reconstruction MSE).

    The quantization unit is a group of `sub_dim` adjacent input channels assigned atomically to
    one centroid, so per-column sequential quantization inside a group is unavailable — this is
    block-GPTQ with block = group. At `sub_dim=1` it reduces to textbook GPTQ.

    Each round refits the codebook on the *previous* round's compensated weights but replays the
    compensation pass from the original weights; compounding the correction would apply it
    repeatedly and diverge. Only `codes` and the final `codebook` are kept, so the footprint is
    identical to an uncompensated encode."""
    out, in_ = weight.shape
    if in_ % sub_dim:
        raise ValueError(f"in_features {in_} not divisible by sub_dim {sub_dim}")
    groups = in_ // sub_dim
    u = hinv_chol.to(device=weight.device, dtype=torch.float32)
    w_fit, errors = weight, []
    codes = torch.empty(out, groups, dtype=torch.long, device=weight.device)

    for _ in range(rounds):
        codebook = pq_quantize(w_fit, sub_dim, n_centroids, iters, max_fit=max_fit)[1].float()
        work = weight.float().clone()
        for g in range(groups):
            lo, hi = g * sub_dim, (g + 1) * sub_dim
            codes[:, g] = assign(work[:, lo:hi], codebook)
            if compensate and hi < in_:
                err = work[:, lo:hi] - codebook[codes[:, g]]
                delta = err @ torch.linalg.inv(u[lo:hi, lo:hi])
                work[:, hi:] -= delta @ u[lo:hi, hi:]
        errors.append(float((codebook[codes].reshape(out, in_) - weight).pow(2).mean()))
        w_fit = work
    return codes, codebook.to(weight.dtype), errors


def compensated_quantize_fused(
    weight: torch.Tensor,
    sub_dim: int,
    n_centroids: int,
    hinv_chol: torch.Tensor,
    iters: int = 10,
    max_fit: int | None = None,
    rounds: int = 3,
    compensate: bool = True,
) -> list[float]:
    """`compensated_quantize` over a fused (n_experts, out, in) tensor, writing the
    reconstruction back in place. Every expert in a layer shares `H` and the group order, so the
    group loop runs once and assignment/compensation batch across experts — ~20k batched steps
    for the whole model instead of 5.2M Python iterations.

    Codebooks stay per-expert (as elsewhere in this codebase), so assignment is a batched cdist
    against each expert's own centroids."""
    n_experts, out, in_ = weight.shape
    groups = in_ // sub_dim
    u = hinv_chol.to(device=weight.device, dtype=torch.float32)
    original = weight.float().clone()
    w_fit, errors = original, []
    codes = torch.empty(n_experts, out, groups, dtype=torch.long, device=weight.device)

    for _ in range(rounds):
        books = torch.stack([
            pq_quantize(w_fit[e], sub_dim, n_centroids, iters, max_fit=max_fit)[1].float()
            for e in range(n_experts)
        ])                                                   # (n_experts, k, sub_dim)
        work = original.clone()
        for g in range(groups):
            lo, hi = g * sub_dim, (g + 1) * sub_dim
            codes[:, :, g] = torch.cdist(work[:, :, lo:hi], books).argmin(dim=2)
            if compensate and hi < in_:
                block_recon = torch.gather(
                    books, 1, codes[:, :, g].unsqueeze(-1).expand(-1, -1, sub_dim))
                delta = (work[:, :, lo:hi] - block_recon) @ torch.linalg.inv(u[lo:hi, lo:hi])
                work[:, :, hi:] -= delta @ u[lo:hi, hi:]
        full_recon = torch.gather(
            books, 1, codes.reshape(n_experts, -1, 1).expand(-1, -1, sub_dim)
        ).reshape(n_experts, out, in_)
        errors.append(float((full_recon - original).pow(2).mean()))
        w_fit = work

    weight.copy_(full_recon.to(weight.dtype))
    return errors
