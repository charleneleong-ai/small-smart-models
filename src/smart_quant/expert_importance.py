"""Profile per-expert activation frequency to drive importance-aware bit allocation.

The idea: hook every MoE router, accumulate how often each expert is selected across a
calibration set, then allocate more bits to hot experts and fewer to cold ones while
holding the average bpw fixed. This gives a codebook quant the same "spend bits where
they're used" advantage that importance-matrix GGUF gets structurally.
"""
from __future__ import annotations

import torch
from torch import nn


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


def bits_from_frequency(
    freq: torch.Tensor,
    avg_bits: float,
    lo: float = 1.5,
    hi: float = 3.0,
) -> torch.Tensor:
    """Per-expert bit budget in [lo, hi], increasing with usage, whose usage-weighted mean
    is exactly `avg_bits` — so the footprint lands on target. Water-fills the hottest
    experts up to `hi` first; `avg_bits` is clamped into [lo, hi] if the mean is infeasible."""
    freq = freq / freq.sum().clamp(min=1e-9)
    bits = torch.full_like(freq, float(lo))
    surplus = min(max(avg_bits, lo), hi) - lo  # usage-weighted bits left to distribute
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
    return bits
