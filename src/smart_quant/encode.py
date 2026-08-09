"""Fake-quantize MoE expert weights with product quantization + per-expert bit allocation.

transformers >=5 stores experts as fused `(num_experts, out_features, in_features)` tensors —
the standard `nn.Linear` layout, since the forward applies `F.linear(x, W)` = `x @ W.T`. Each
expert is a 2D slice `W[e]`. We product-quantize each slice with a shared codebook at a
per-expert bit budget — uniform, or driven by routing usage via `bits_from_frequency` — then
write the dequantized reconstruction back. This "fake quantization" lets the fp16 model
reflect quantized weights for quality measurement without custom inference kernels.

`codebook_order > 1` splits each expert's budget across that many residual stages
(second-order codebooks); `codebook_order=1` is a single-stage product quantization.

`channel_weight` applies activation-derived importance: sub-vectors run along the input dim, so
each sub-vector's coordinates carry different weights and the codebook fit minimizes
sum_ij w_j (W_ij - West_ij)^2. Weights steer the fit only — reconstruction is still a plain
codebook lookup, so nothing extra is stored and the footprint is unchanged.
"""
from __future__ import annotations

import torch

from smart_quant.codebook import pq_bpw, pq_dequantize, residual_pq_quantize
from smart_quant.compensate import compensated_quantize_fused, damped_inverse
# layer_index lives with the profiler that emits keys with it; re-exported here for callers.
from smart_quant.expert_importance import bits_from_frequency, layer_index
from smart_quant.lattice import quantize_e8_fused

__all__ = ["layer_index", "centroids_for_bits", "quantize_fused_experts", "quantize_experts",
           "noise_inject_experts"]


def centroids_for_bits(bits: float, sub_dim: int, lo: int = 16, hi: int = 4096) -> int:
    """Codebook size realizing ~`bits` bpw at `sub_dim` (bpw ~ log2(n)/sub_dim), snapped to a
    power of two and clamped to [lo, hi]."""
    return int(min(max(2 ** round(bits * sub_dim), lo), hi))


def quantize_fused_experts(
    weight: torch.Tensor, bits_per_expert: torch.Tensor, sub_dim: int, iters: int = 10,
    codebook_order: int = 1, channel_weight: torch.Tensor | None = None,
    hinv_chol: torch.Tensor | None = None, rounds: int = 3, compensate: bool = True,
) -> tuple[float, int]:
    """Fake-quantize a fused (num_experts, out_features, in_features) weight in place along the
    expert dim, each expert at its own `bits_per_expert` budget. With `codebook_order > 1` the
    budget is split evenly across that many residual stages. Returns (realized_bits, n_weights):
    total stored bits (indices + shared codebooks, via `pq_bpw`) and the weight count.

    `channel_weight` is per-input-channel importance, (num_experts, in_) — the caller normalizes
    a layer-granularity vector by expanding it, so this loop never shape-dispatches.

    `hinv_chol` enables GPTQ-style error compensation across this tensor's input dim. It changes
    which codes are chosen, never the bit count, so the returned footprint is identical to the
    uncompensated encode. Requires `codebook_order == 1` and no `channel_weight` — compensation
    and the Phase-7 weighting are separate techniques and are not combined."""
    if hinv_chol is not None:
        if codebook_order != 1 or channel_weight is not None:
            raise ValueError("compensation supports codebook_order=1 without channel_weight")
        if not bool((bits_per_expert == bits_per_expert[0]).all()):
            raise ValueError(
                "compensation requires uniform bits_per_expert — batching over experts needs one "
                "codebook size, so per-expert allocation and compensation cannot be combined")
        k = centroids_for_bits(float(bits_per_expert[0]), sub_dim)
        compensated_quantize_fused(weight, sub_dim, k, hinv_chol, iters=iters,
                                   max_fit=max(4096, k * 8), rounds=rounds,
                                   compensate=compensate)
        out, in_ = weight.shape[1], weight.shape[2]
        n_experts = weight.shape[0]
        return pq_bpw(out, in_, sub_dim, [k]) * out * in_ * n_experts, out * in_ * n_experts

    realized_bits, n_weights = 0.0, 0
    for e in range(weight.shape[0]):
        k = centroids_for_bits(float(bits_per_expert[e]) / codebook_order, sub_dim)
        stage_centroids = [k] * codebook_order  # even split — every stage same codebook size
        max_fit = max(4096, k * 8)
        cw = None if channel_weight is None else channel_weight[e]
        codes, codebooks = residual_pq_quantize(
            weight[e], sub_dim, stage_centroids, iters=iters, max_fit=max_fit, channel_weight=cw)
        weight[e] = pq_dequantize(codes, codebooks).to(weight.dtype)
        out, in_ = weight[e].shape
        realized_bits += pq_bpw(out, in_, sub_dim, stage_centroids) * out * in_
        n_weights += out * in_
    return realized_bits, n_weights


def quantize_experts(
    model, avg_bits: float, sub_dim: int = 4, freqs: dict | None = None, iters: int = 10,
    bits_lo: float = 1.5, bits_hi: float = 3.0, codebook_order: int = 1,
    importance: dict[str, torch.Tensor] | None = None,
    hessians: dict[int, torch.Tensor] | None = None, rounds: int = 3,
    compensate: bool = True, lattice: bool = False,
) -> list[dict]:
    """Walk a model's fused MoE expert modules and fake-quantize them. With `freqs` (router
    name -> usage frequencies), per-expert bits are water-filled to `avg_bits` by usage within
    [bits_lo, bits_hi]; otherwise every expert gets `avg_bits`.

    `lattice=True` quantizes to the E8 lattice instead of a learned codebook. `avg_bits` then acts
    as a *target* rate realized by per-tensor scale calibration, and `freqs`/`importance` are
    ignored — a lattice takes no calibration statistic and has no per-expert codebook to weight.

    `importance` maps
    `"<layer_index>.<param_name>"` to per-input-channel weights, keyed per *parameter* because
    `gate_up_proj` and `down_proj` have different `in_features`, and by layer *index* because the
    profiler and the encode load the model through different wrappers. A supplied `importance`
    that matches nothing raises rather than silently degrading to an unweighted encode — that
    failure would otherwise be published as a null result. Returns per-layer stats.

    `hessians` maps layer index to that layer's input second moment. Compensation is applied to
    `gate_up_proj` only — `down_proj`'s intermediate is expert-specific and nearly isotropic, so
    it stays uniform and serves as the phase's control."""
    freq_by_layer = {layer_index(k): v for k, v in freqs.items()} if freqs else {}
    matched_importance = 0
    matched_hessians = 0
    stats = []
    for name, module in model.named_modules():
        if not type(module).__name__.endswith("Experts"):
            continue
        fused = [(pn, p) for pn, p in module.named_parameters(recurse=False) if p.dim() == 3]
        if not fused:
            continue
        n_experts = fused[0][1].shape[0]
        freq = freq_by_layer.get(layer_index(name))
        if freq is not None and len(freq) == n_experts:
            bits = bits_from_frequency(freq, avg_bits, lo=bits_lo, hi=bits_hi)
        else:
            bits = torch.full((n_experts,), float(avg_bits))
        layer_bits, layer_weights = 0.0, 0
        with torch.no_grad():
            for param_name, weight in fused:
                cw = None
                if importance is not None:
                    cw = importance.get(f"{layer_index(name)}.{param_name}")
                    if cw is not None:
                        matched_importance += 1
                        if cw.dim() == 1:  # layer granularity — one vector for every expert
                            cw = cw.expand(n_experts, -1)
                if lattice:
                    # the lattice has no fitted codebook, so neither error compensation nor
                    # channel weighting applies — both are corrections to a k-means fit
                    fused_bits, fused_weights = quantize_e8_fused(weight, avg_bits)
                else:
                    hin = None
                    if hessians is not None and param_name == "gate_up_proj":
                        h = hessians.get(layer_index(name))
                        if h is not None:
                            hin = damped_inverse(h)
                            matched_hessians += 1
                    fused_bits, fused_weights = quantize_fused_experts(
                        weight, bits, sub_dim, iters, codebook_order=codebook_order,
                        channel_weight=cw, hinv_chol=hin, rounds=rounds, compensate=compensate)
                layer_bits += fused_bits
                layer_weights += fused_weights
        stats.append({"layer": name, "n_experts": n_experts,
                      "bits_min": round(float(bits.min()), 2),
                      "bits_max": round(float(bits.max()), 2),
                      "quant_bits": layer_bits, "quant_weights": layer_weights})
    if importance is not None and not matched_importance:
        raise KeyError(
            "importance supplied but no key matched any fused expert parameter. "
            "Expected keys '<layer_index>.<param_name>', e.g. '0.gate_up_proj'.")
    if hessians is not None and not matched_hessians:
        raise KeyError(
            "hessians supplied but no key matched any gate_up_proj layer. "
            "Expected integer layer-index keys, e.g. 0.")
    return stats


def noise_inject_experts(
    model, avg_bits: float, sub_dim: int = 4, iters: int = 10,
) -> list[dict]:
    """Inject Gaussian noise into MoE expert weights, matched to PQ's perturbation RMS.

    For each expert, this:
    1. Fits a PQ codebook and measures the RMS of the quantization error (weight - recon)
    2. Restores the original weight
    3. Injects Gaussian noise with std = error_rms

    This produces unstructured noise at the same magnitude as PQ, testing whether
    regularization comes from noise magnitude (generic) or noise structure (learned codebook)."""
    from smart_quant.codebook import pq_dequantize, residual_pq_quantize

    stats = []
    for name, module in model.named_modules():
        if not type(module).__name__.endswith("Experts"):
            continue
        fused = [(pn, p) for pn, p in module.named_parameters(recurse=False) if p.dim() == 3]
        if not fused:
            continue
        n_experts = fused[0][1].shape[0]
        bits = torch.full((n_experts,), float(avg_bits))
        layer_bits, layer_weights = 0.0, 0
        with torch.no_grad():
            for param_name, weight in fused:
                # Step 1: PQ quantize to measure the error RMS
                codes, codebooks = residual_pq_quantize(
                    weight[0].float(), sub_dim, [centroids_for_bits(avg_bits, sub_dim)],
                    iters=iters, max_fit=max(4096, 256))
                recon = pq_dequantize(codes, codebooks).to(weight.dtype)
                error_rms = (weight[0].float() - recon.float()).pow(2).mean().sqrt().item()

                # Step 2: Inject Gaussian noise with matched RMS to all experts
                noise = torch.randn_like(weight.float()) * error_rms
                weight.copy_((weight.float() + noise).to(weight.dtype))

                out, in_ = weight.shape[1], weight.shape[2]
                layer_bits += pq_bpw(out, in_, sub_dim, [centroids_for_bits(avg_bits, sub_dim)]) * out * in_ * n_experts
                layer_weights += out * in_ * n_experts
        stats.append({"layer": name, "n_experts": n_experts,
                      "bits_min": round(avg_bits, 2), "bits_max": round(avg_bits, 2),
                      "quant_bits": layer_bits, "quant_weights": layer_weights,
                      "noise_rms": error_rms})
    return stats
