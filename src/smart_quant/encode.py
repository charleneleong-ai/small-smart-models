"""Fake-quantize MoE expert weights with product quantization + per-expert bit allocation.

transformers >=5 stores experts as fused `(num_experts, d_in, d_out)` tensors, so each
expert is a 2D slice `W[e]`. We product-quantize each slice with a shared codebook at a
per-expert bit budget — uniform, or driven by routing usage via `bits_from_frequency` — then
write the dequantized reconstruction back. This "fake quantization" lets the fp16 model
reflect quantized weights for quality measurement without custom inference kernels.
"""
from __future__ import annotations

import re

import torch

from smart_quant.codebook import pq_dequantize, pq_quantize
from smart_quant.expert_importance import bits_from_frequency

__all__ = ["layer_index", "centroids_for_bits", "quantize_fused_experts", "quantize_experts"]


def layer_index(name: str) -> int | None:
    """Extract the decoder-layer index from a module/param name (`...layers.<N>...`), so
    routers and experts can be matched by layer regardless of model-wrapper name prefixes."""
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def centroids_for_bits(bits: float, sub_dim: int, lo: int = 16, hi: int = 4096) -> int:
    """Codebook size realizing ~`bits` bpw at `sub_dim` (bpw ~ log2(n)/sub_dim), snapped to a
    power of two and clamped to [lo, hi]."""
    return int(min(max(2 ** round(bits * sub_dim), lo), hi))


def quantize_fused_experts(
    weight: torch.Tensor, bits_per_expert: torch.Tensor, sub_dim: int, iters: int = 10
) -> torch.Tensor:
    """Fake-quantize a fused (num_experts, d_in, d_out) weight in place along the expert dim,
    each expert at its own `bits_per_expert` budget."""
    for e in range(weight.shape[0]):
        n_centroids = centroids_for_bits(float(bits_per_expert[e]), sub_dim)
        # fit the codebook on a bounded subsample (>= a healthy multiple of the codebook
        # size) so k-means stays fast on large expert weights; all sub-vectors still assigned
        max_fit = max(4096, n_centroids * 8)
        codes, codebook = pq_quantize(weight[e], sub_dim, n_centroids, iters=iters, max_fit=max_fit)
        weight[e] = pq_dequantize(codes, codebook).to(weight.dtype)
    return weight


def quantize_experts(
    model, avg_bits: float, sub_dim: int = 4, freqs: dict | None = None, iters: int = 10,
    bits_lo: float = 1.5, bits_hi: float = 3.0,
) -> list[dict]:
    """Walk a model's fused MoE expert modules and fake-quantize them. With `freqs` (router
    name -> usage frequencies), per-expert bits are water-filled to `avg_bits` by usage within
    [bits_lo, bits_hi]; otherwise every expert gets `avg_bits`. Returns per-layer stats."""
    freq_by_layer = {layer_index(k): v for k, v in freqs.items()} if freqs else {}
    stats = []
    for name, module in model.named_modules():
        if not type(module).__name__.endswith("Experts"):
            continue
        fused = [p for _, p in module.named_parameters(recurse=False) if p.dim() == 3]
        if not fused:
            continue
        n_experts = fused[0].shape[0]
        freq = freq_by_layer.get(layer_index(name))
        if freq is not None and len(freq) == n_experts:
            bits = bits_from_frequency(freq, avg_bits, lo=bits_lo, hi=bits_hi)
        else:
            bits = torch.full((n_experts,), float(avg_bits))
        with torch.no_grad():
            for weight in fused:
                quantize_fused_experts(weight, bits, sub_dim, iters)
        stats.append({"layer": name, "n_experts": n_experts,
                      "bits_min": round(float(bits.min()), 2),
                      "bits_max": round(float(bits.max()), 2)})
    return stats
