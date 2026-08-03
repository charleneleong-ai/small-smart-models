"""Per-row affine scalar quantization — the rate-limited family for the allocation 2×2.

Product quantization is *fit*-limited: each expert's shared codebook is fitted by Lloyd on a
finite sample, so realized error saturates below what the index rate promises. A per-row scalar
quantizer has no global fit — `scale = (max - min) / levels` is closed-form per row — so its
error tracks the rate directly (~6 dB/bit). The two families are the mechanism axis of the
allocation experiment: importance allocation can only move error where a quantizer is
rate-limited.
"""
from __future__ import annotations

import torch


def scalar_quantize(weight: torch.Tensor, bits: float) -> torch.Tensor:
    """Per-row min-max affine scalar quantize of a (rows, cols) weight, returning the
    reconstruction. `bits` is the per-element rate, realized as `round(2**bits)` levels per
    row (clamped to at least 2); the two per-row scale params amortise to ~0 bits/element at
    realistic `cols`. A constant row reproduces exactly."""
    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits}")
    levels = max(2, round(2**bits))
    lo = weight.min(dim=-1, keepdim=True).values
    hi = weight.max(dim=-1, keepdim=True).values
    span = (hi - lo).clamp_min(torch.finfo(weight.dtype).tiny)
    q = torch.round((weight - lo) / span * (levels - 1))
    return lo + q / (levels - 1) * span
