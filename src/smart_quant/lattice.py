"""E8 lattice quantization.

A stored codebook costs O(2^{kd} * d) to keep *and* to search — exponential in both bitrate and
dimension. Our shape sweep measured that wall directly: k-means at sub_dim=8 with k=65536 costs
6.0 bpw, 4.0 of which is the codebook itself, and still loses to sub_dim=2 at 3.0 bpw. A lattice
escapes it because its points are computed rather than tabulated, which makes sub_dim=8 reachable
at zero storage.

E8 is the densest sphere packing in 8 dimensions and is what QuIP# uses. Nearest-point search is
Conway & Sloane's: E8 = D8 union (D8 + 1/2), and the nearest D8 point is coordinate-wise rounding
with a parity fix — O(d) per sub-vector, no search.

This module supports *fake* quantization only: the reconstruction is written back and codes are
never stored, so no canonical shell enumeration or encoder is needed. Those are deployment
concerns. Rate is accounted from the measured number of distinct points actually used.
"""
from __future__ import annotations

import math

import torch

__all__ = ["nearest_e8", "distinct_points", "calibrate_scale", "quantize_e8_fused"]

# Prime, comfortably larger than any doubled coordinate magnitude these scales produce.
HASH_BASE = 40507


def nearest_d8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of D8 (integer coordinates summing to an even number).

    Round coordinate-wise; if the sum comes out odd, re-round the single coordinate whose rounding
    was least confident, which is the cheapest way back onto the even-sum sublattice."""
    r = torch.round(x)
    odd = r.sum(1) % 2 != 0
    if odd.any():
        err = x - r
        j = err.abs().argmax(dim=1)
        rows = torch.arange(x.shape[0], device=x.device)
        flip = torch.sign(err[rows, j])
        flip[flip == 0] = 1.0
        r[rows[odd], j[odd]] += flip[odd]
    return r


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Nearest E8 point to each row of x (n, 8). E8 is D8 union its half-shift, so quantize to
    both cosets and keep whichever came out closer."""
    a = nearest_d8(x)
    b = nearest_d8(x - 0.5) + 0.5
    closer = (x - a).pow(2).sum(1) <= (x - b).pow(2).sum(1)
    return torch.where(closer.unsqueeze(1), a, b)


def distinct_points(pts: torch.Tensor) -> int:
    """Number of distinct lattice points, via an integer polynomial hash.

    Coordinates are half-integers, so doubling makes them integral. `unique(dim=0)` over tens of
    millions of rows is not viable; a 1-D unique over hashes is."""
    key = (pts * 2).to(torch.int64)
    # Horner form, deliberately allowed to wrap: HASH_BASE**7 is ~1e32 and overflows int64, so a
    # precomputed power vector is not an option. Wrapping is a polynomial hash mod 2**64.
    h = torch.zeros(pts.shape[0], dtype=torch.int64, device=pts.device)
    for i in range(pts.shape[1]):
        h = h * HASH_BASE + key[:, i]
    return int(torch.unique(h).numel())


def calibrate_scale(pool: torch.Tensor, target_bpw: float, sub_dim: int = 8,
                    iters: int = 24, max_fit: int = 2_000_000) -> float:
    """Scale whose distinct-point count realizes `target_bpw`.

    Distinct count decreases monotonically as the scale coarsens, so bisection converges — in the
    geometric mean, since usable scales span orders of magnitude. Calibrated on a strided
    subsample for speed; the caller re-measures the realized rate on the full tensor, so a
    subsample that lands off-target shows up in the reported bpw rather than silently."""
    fit = pool if pool.shape[0] <= max_fit else pool[
        torch.linspace(0, pool.shape[0] - 1, max_fit).round().long()]
    target_points = 2.0 ** (target_bpw * sub_dim)
    lo, hi = 1e-5, 1.0
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if distinct_points(nearest_e8(fit / mid)) > target_points:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def quantize_e8_fused(weight: torch.Tensor, target_bpw: float,
                      sub_dim: int = 8) -> tuple[float, int]:
    """Fake-quantize a fused (num_experts, out, in) weight to the E8 lattice, in place.

    One scale serves the whole tensor: that is what the pooled measurement used, and it already
    beat 32 individually fitted per-expert k-means codebooks. Returns (realized_bits, n_weights)
    with **no codebook term** — lattice points are computed, so the only costs are the index and a
    single fp16 scale."""
    if sub_dim != 8:
        raise ValueError("E8 is defined for sub_dim=8")
    if weight.shape[-1] % sub_dim:
        raise ValueError(f"in_features {weight.shape[-1]} not divisible by sub_dim {sub_dim}")

    pool = weight.reshape(-1, sub_dim).float()
    scale = calibrate_scale(pool, target_bpw, sub_dim)
    pts = nearest_e8(pool / scale)
    weight.copy_((pts * scale).reshape(weight.shape).to(weight.dtype))

    index_bits = math.ceil(math.log2(max(distinct_points(pts), 2)))
    n_weights = weight.numel()
    return index_bits * (n_weights / sub_dim) + 16, n_weights
