from __future__ import annotations

import torch

from smart_quant.codebook import assign, pq_quantize

__all__ = ["damped_inverse", "compensated_quantize"]


def damped_inverse(h: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor damped `H^-1`, in float64.

    `damp * mean(diag(H))` on diagonal GPTQ's standard preconditioning.
    load-bearing here, not decorative: layer 0's covariance condition ~1e5
    per-expert estimate over few routed tokens outright rank-deficient. Dead input channels
    (all-zero column, real experts get unit diagonal contribute no
    compensation non-positive-definite matrix."""
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
