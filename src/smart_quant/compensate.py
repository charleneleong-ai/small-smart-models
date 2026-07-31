from __future__ import annotations

import torch

from smart_quant.codebook import assign, pq_quantize

__all__ = ["damped_inverse", "compensated_quantize"]


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
