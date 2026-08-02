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


def point_hash(pts: torch.Tensor) -> torch.Tensor:
    """Collision-resistant int64 key per lattice point, so distinct counts can be accumulated
    across chunks without holding every point in memory.

    Coordinates are half-integers, so doubling makes them integral. Horner form, deliberately
    allowed to wrap: HASH_BASE**7 is ~1e32 and overflows int64, so a precomputed power vector is
    not an option — wrapping makes this a polynomial hash mod 2**64."""
    key = (pts * 2).to(torch.int64)
    h = torch.zeros(pts.shape[0], dtype=torch.int64, device=pts.device)
    for i in range(pts.shape[1]):
        h = h * HASH_BASE + key[:, i]
    return h


def distinct_points(pts: torch.Tensor) -> int:
    """Number of distinct lattice points. `unique(dim=0)` over tens of millions of rows is not
    viable; a 1-D unique over hashes is."""
    return int(torch.unique(point_hash(pts)).numel())


HEADROOM = 4  # sub-vectors per addressable point before the fit degenerates into memorisation


def strided_indices(n: int, k: int, device: torch.device) -> torch.Tensor:
    """`k` evenly spaced indices into a length-`n` axis, in exact integer arithmetic.

    `torch.linspace` returns float32, whose 24-bit mantissa cannot represent indices above 2**24.
    A pooled expert tensor has ~67M sub-vectors, so the final index 2**26 - 1 rounds *up* to 2**26
    and gathers out of bounds. `codebook.py` uses linspace safely only because it subsamples
    per-expert, well under the limit."""
    if k >= n:
        return torch.arange(n, device=device)
    return torch.arange(k, device=device, dtype=torch.long) * (n - 1) // (k - 1)


def distinct_at_scale(pool: torch.Tensor, scale: float, chunk: int = 4_000_000) -> int:
    """Distinct lattice points over the whole pool at `scale`, chunked so a 67M-row tensor does
    not build its rounding temporaries all at once."""
    hashes = [point_hash(nearest_e8(pool[i:i + chunk].float() / scale))
              for i in range(0, pool.shape[0], chunk)]
    return int(torch.unique(torch.cat(hashes)).numel())


def calibrate_scale(pool: torch.Tensor, target_bpw: float, sub_dim: int = 8,
                    iters: int = 24, max_fit: int = 8_000_000, refine: int = 8) -> float:
    """Scale whose distinct-point count realizes `target_bpw`.

    Distinct count decreases monotonically as the scale coarsens, so bisection converges — in the
    geometric mean, since usable scales span orders of magnitude. Calibrated on a strided
    subsample for speed; the caller re-measures the realized rate on the full tensor, so a
    subsample that lands off-target shows up in the reported bpw rather than silently.

    Raises when the pool cannot support the target. Without that guard the bisection never sees a
    count above target, drives the scale to its floor, and hands back a near-lossless fit in which
    almost every sub-vector has its own code — memorisation, not quantisation, reported at a
    fictional rate. That degeneracy is exactly what invalidated an early 2^20 measurement."""
    target_points = 2.0 ** (target_bpw * sub_dim)
    if pool.shape[0] < HEADROOM * target_points:
        raise ValueError(
            f"{pool.shape[0]} sub-vectors cannot realize {target_bpw} bpw: that needs "
            f"{target_points:.0f} addressable points and at least {HEADROOM}x as many "
            f"sub-vectors. Use a larger tensor or a lower target.")

    fit = pool[strided_indices(pool.shape[0], max_fit, pool.device)].float()
    lo, hi = 1e-5, 1.0
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if distinct_points(nearest_e8(fit / mid)) > target_points:
            lo = mid
        else:
            hi = mid

    if refine and fit.shape[0] < pool.shape[0]:
        # A subsample systematically undercounts distinct points, so the coarse scale realizes a
        # *higher* rate than asked — 2.75 against a 2.5 target on a real tensor. Re-bisect against
        # the full pool inside a widened bracket. Matched footprint is the comparison's whole
        # basis, so paying a few seconds here is cheaper than a mismatched encode.
        lo, hi = lo * 0.5, hi * 2.0
        for _ in range(refine):
            mid = (lo * hi) ** 0.5
            if distinct_at_scale(pool, mid) > target_points:
                lo = mid
            else:
                hi = mid
    return (lo * hi) ** 0.5


def quantize_e8_fused(weight: torch.Tensor, target_bpw: float, sub_dim: int = 8,
                      chunk: int = 4_000_000) -> tuple[float, int]:
    """Fake-quantize a fused (num_experts, out, in) weight to the E8 lattice, in place.

    One scale serves the whole tensor: that is what the pooled measurement used, and it already
    beat 32 individually fitted per-expert k-means codebooks. Returns (realized_bits, n_weights)
    with **no codebook term** — lattice points are computed, so the only costs are the index and a
    single fp16 scale.

    Quantization runs in chunks. A real `gate_up_proj` holds ~67M sub-vectors, and rounding them
    all at once builds >10 GB of temporaries on top of a 67 GB model — the shape of failure that
    OOM'd Phase 8. Chunking is exact: sub-vectors are independent given the tensor's single
    scale."""
    if sub_dim != 8:
        raise ValueError("E8 is defined for sub_dim=8")
    if weight.shape[-1] % sub_dim:
        raise ValueError(f"in_features {weight.shape[-1]} not divisible by sub_dim {sub_dim}")

    pool = weight.reshape(-1, sub_dim)          # a view: writes land back in `weight`
    scale = calibrate_scale(pool, target_bpw, sub_dim)

    hashes = []
    for i in range(0, pool.shape[0], chunk):
        pts = nearest_e8(pool[i:i + chunk].float() / scale)
        pool[i:i + chunk] = (pts * scale).to(weight.dtype)
        hashes.append(point_hash(pts))
    distinct = int(torch.unique(torch.cat(hashes)).numel())

    index_bits = math.ceil(math.log2(max(distinct, 2)))
    n_weights = weight.numel()
    return index_bits * (n_weights / sub_dim) + 16, n_weights
