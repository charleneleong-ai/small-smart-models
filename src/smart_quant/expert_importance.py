"""Profile per-expert activation frequency to drive importance-aware bit allocation.

The idea: hook every MoE router, accumulate how often each expert is selected across a
calibration set, then allocate more bits to hot experts and fewer to cold ones while
holding the average bpw fixed. This gives a codebook quant the same "spend bits where
they're used" advantage that importance-matrix GGUF gets structurally.
"""
from __future__ import annotations

import re
from typing import Callable

import torch
from torch import nn


def layer_index(name: str) -> int | None:
    """Extract the decoder-layer index from a module/param name (`...layers.<N>...`), so
    routers, experts and calibration artifacts can be matched by layer regardless of
    model-wrapper name prefixes — `AutoModel` and `AutoModelForCausalLM` differ here."""
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


class ExpertUsageProfiler:
    """Counts expert selections per MoE layer via forward hooks on the routers.

    The default `router_predicate` matches each MoE block's `gate` submodule by name, which
    holds across the transformers router refactor (<=4 exposes an `nn.Linear` gate, >=5 a
    `Qwen3MoeTopKRouter`) and across the multimodal nesting
    (`model.language_model.layers.{i}.mlp.gate` in Qwen3.6-35B-A3B). The hook reads the
    router logits — element 0 when the router returns a tuple — and top-k's them to recover
    each token's expert selection.
    """

    def __init__(self, model: nn.Module, top_k: int, num_experts: int, router_predicate=None):
        self.model = model
        self.top_k = top_k
        self.num_experts = num_experts
        self.counts: dict[str, torch.Tensor] = {}
        self.predicate = router_predicate or self.default_predicate
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def default_predicate(self, name: str, module: nn.Module) -> bool:
        # The router is the MoE block's `gate`. transformers >=5 makes it a custom
        # Qwen3MoeTopKRouter (not an nn.Linear); <=4 makes it nn.Linear(hidden, num_experts).
        # Match by name across both, with an out_features fallback. `...mlp.shared_expert_gate`
        # does not end in "mlp.gate", so the shared-expert gate is excluded.
        if name.endswith("mlp.gate"):
            return True
        return isinstance(module, nn.Linear) and module.out_features == self.num_experts

    def _make_hook(self, name: str):
        def hook(_module, _inp, output):
            logits = output[0] if isinstance(output, tuple) else output
            chosen = logits.topk(self.top_k, dim=-1).indices.reshape(-1)
            n_experts = logits.shape[-1]
            hist = torch.bincount(chosen.cpu(), minlength=n_experts)
            prev = self.counts.get(name)
            self.counts[name] = hist if prev is None else prev + hist
        return hook

    def __enter__(self) -> "ExpertUsageProfiler":
        for name, module in self.model.named_modules():
            if self.predicate(name, module):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def frequencies(self) -> dict[str, torch.Tensor]:
        """Normalized selection frequency per expert, per layer."""
        return {name: c / c.sum().clamp(min=1) for name, c in self.counts.items()}


class ActivationImportanceProfiler:
    """Accumulates E[x^2] of each fused expert projection's input over a calibration set.

    One forward-pre hook per `Experts` module suffices: transformers >=5 passes the routing in as
    `forward(hidden_states, top_k_index, top_k_weights)`, so there is no router hook to pair with.

    Keys are `"<layer_index>.<param_name>"` — the layer *index*, never the module path, so the
    artifact survives the different wrapper prefixes `AutoModel` (the profiler) and
    `AutoModelForCausalLM` (the encode) put on the same modules.

    Arch coupling is deliberately confined to `make_hook`: fused params named
    `gate_up_proj`/`down_proj`, and a SwiGLU packed **gate-first** so `chunk(2, -1)` splits it.
    The name assumptions fail loudly; the packing assumption would silently mis-attribute the
    `down_proj` statistic on a model that packed up-first.

    Sums are accumulated per-expert on-device (only ~130 MB across a 40x256 model) and moved to
    CPU once in `importance()`; per-expert host syncs in the hook cost more than the statistic.
    `granularity` is a read-time argument, so one calibration pass serves both arms.
    """

    def __init__(self, model: nn.Module, num_experts: int):
        self.model = model
        self.num_experts = num_experts
        self.sumsq: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(self, key: str, expert: int, x: torch.Tensor) -> None:
        """Fold x (n_tokens, d_in) into the running on-device sum for one expert."""
        sq = x.detach().float().pow(2).sum(0)
        if key not in self.sumsq:
            self.sumsq[key] = torch.zeros(self.num_experts, sq.shape[0], device=sq.device)
            self.counts[key] = torch.zeros(self.num_experts, device=sq.device)
        self.sumsq[key][expert] += sq
        self.counts[key][expert] += x.shape[0]

    def make_hook(self, name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        layer = layer_index(name)
        prefix = f"{layer}." if layer is not None else ""

        def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            hidden, top_k_index = args[0], args[1]
            flat = hidden.reshape(-1, hidden.shape[-1])
            # one host sync per layer for the expert list; everything else stays on device
            for e in top_k_index.unique().tolist():
                token_idx = (top_k_index == e).any(dim=-1).nonzero(as_tuple=True)[0]
                x_e = flat[token_idx]
                self.add(f"{prefix}gate_up_proj", e, x_e)
                gate, up = nn.functional.linear(x_e, module.gate_up_proj[e]).chunk(2, dim=-1)
                self.add(f"{prefix}down_proj", e, module.act_fn(gate) * up)
        return hook

    def __enter__(self) -> "ActivationImportanceProfiler":
        for name, module in self.model.named_modules():
            if type(module).__name__.endswith("Experts"):
                self.handles.append(module.register_forward_pre_hook(self.make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def importance(self, granularity: str = "expert") -> dict[str, torch.Tensor]:
        """Mean x^2 per key, on CPU: (n_experts, d_in) per expert, (d_in,) marginalized for a
        layer. The layer statistic is the token-weighted marginal of the per-expert one, so the
        two arms differ only in that marginalization."""
        out: dict[str, torch.Tensor] = {}
        for key, total in self.sumsq.items():
            total, cnt = total.cpu(), self.counts[key].cpu()
            out[key] = (total.sum(0) / cnt.sum().clamp(min=1.0) if granularity == "layer"
                        else total / cnt.clamp(min=1.0).unsqueeze(1))
        return out


def normalize_importance(w: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Scale each importance vector to mean 1, optionally compressing its dynamic range first.
    The weighted fit is invariant to the absolute scale, so this is about comparability across
    experts; `alpha < 1` is the lever for when unmoderated magnitudes let a few hot channels
    capture every centroid."""
    if alpha != 1.0:
        w = w.clamp(min=0).pow(alpha)
    return w / w.mean(dim=-1, keepdim=True).clamp(min=1e-12)


def shrink_importance(
    raw: torch.Tensor, counts: torch.Tensor, layer_stat: torch.Tensor, tau: float = 1000.0,
) -> torch.Tensor:
    """Empirical-Bayes shrink per-expert E[x^2] (n_experts, d_in) toward the layer statistic
    (d_in,) by routed-token count. At top-8 of 256 the tail experts see few tokens, so their raw
    statistic is mostly noise; `tau` is the pseudo-count at which raw and layer contribute
    equally, and a zero-token expert falls back exactly to the layer."""
    n = counts.to(raw.dtype).unsqueeze(1)
    return (n * raw + tau * layer_stat.to(raw.dtype).unsqueeze(0)) / (n + tau)


def arithmetic_centered(bits: torch.Tensor, target: float, lo: float, hi: float) -> torch.Tensor:
    """Shift-and-clamp `bits` so its *arithmetic* (storage) mean hits `target`, preserving the
    [lo, hi] range and the per-expert ordering. Storage cost is sum(bits)/n — the quantity a byte
    budget actually pays — not the usage-weighted mean the water-fill pins."""
    dlo, dhi = lo - hi, hi - lo
    for _ in range(60):
        d = (dlo + dhi) / 2
        if (bits + d).clamp(lo, hi).mean() < target:
            dlo = d
        else:
            dhi = d
    return (bits + dlo).clamp(lo, hi)


def bits_from_frequency(
    freq: torch.Tensor,
    avg_bits: float,
    lo: float = 1.5,
    hi: float = 3.0,
    storage_centered: bool = True,
) -> torch.Tensor:
    """Per-expert bit budget in [lo, hi], increasing with usage, whose *storage* mean
    (sum(bits)/n) is exactly `avg_bits` — so the footprint lands on target. Water-fills the
    hottest experts up to `hi` first; `avg_bits` is clamped into [lo, hi] if infeasible.

    With `storage_centered=False` the raw water-fill is returned, whose *usage-weighted* mean
    lands on target instead. That is not the footprint: skewed routing lets the arithmetic mean
    drift below target, silently shrinking the tensor the encode must match — the Phase-3
    artifact (rows targeted 2.0 bpw but realized ~1.63)."""
    freq = freq / freq.sum().clamp(min=1e-9)
    bits = torch.full_like(freq, float(lo))
    target = min(max(avg_bits, lo), hi)
    surplus = target - lo  # usage-weighted bits left to distribute
    for i in torch.argsort(freq, descending=True):
        if surplus <= 1e-12:
            break
        cap = (hi - lo) * freq[i].item()  # weighted cost to max out expert i
        if surplus >= cap:
            bits[i] = hi
            surplus -= cap
        else:
            bits[i] = lo + surplus / freq[i].clamp(min=1e-9)
            surplus = 0.0
    if not storage_centered:
        return bits
    return arithmetic_centered(bits, target, lo, hi)


class HessianProfiler:
    """Accumulates the input second moment `X^T X` per MoE layer over a calibration set.

    Keyed by layer *index*, never module path, so the artifact survives the different wrapper
    prefixes `AutoModel` (the profiler) and `AutoModelForCausalLM` (the encode) put on the same
    modules — the failure mode that nearly shipped in Phase 7.

    This is `gate_up_proj`'s input covariance: the hidden states entering the block, shared by
    every expert in the layer. `down_proj`'s input is the expert-specific intermediate and has no
    such shortcut; Phase 8 leaves it uniform.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.acc: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(self, layer: int) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            gram = x.T @ x
            self.acc[layer] = gram if layer not in self.acc else self.acc[layer] + gram
            self.counts[layer] = self.counts.get(layer, 0) + x.shape[0]
        return hook

    def __enter__(self) -> "HessianProfiler":
        for name, module in self.model.named_modules():
            layer = layer_index(name)
            if type(module).__name__.endswith("Experts") and layer is not None:
                self.handles.append(module.register_forward_pre_hook(self.make_hook(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def hessians(self) -> dict[int, torch.Tensor]:
        """Mean `x x^T` per layer, on CPU."""
        return {k: (v / max(self.counts[k], 1)).cpu() for k, v in self.acc.items()}
