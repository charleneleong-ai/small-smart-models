import torch
from torch import nn


class TinyExperts(nn.Module):
    """Mirrors the transformers-5 fused Experts contract: (num_experts, out, in) params and a
    forward taking the routing."""

    def __init__(self, num_experts: int = 4, hidden: int = 8, inter: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.1)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return torch.zeros_like(hidden_states)


def wrap_in_layers(experts: nn.Module, prefix_depth: int = 1) -> nn.Module:
    """Nest `experts` at `...layers.0.mlp.experts` under `prefix_depth` wrapper modules, mimicking
    how AutoModel and AutoModelForCausalLM put different prefixes on the same module path."""
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = experts
    node = nn.Module()
    node.layers = nn.ModuleList([block])
    for _ in range(prefix_depth):
        outer = nn.Module()
        outer.inner = node
        node = outer
    return node


def heterogeneous(*shape: int) -> torch.Tensor:
    """Random weights whose input channels have genuinely different scales, as trained weights
    do. Fixtures must not use plain randn: on i.i.d. columns every sub-vector is exchangeable,
    so importance weighting has nothing to exploit and is expected to do nothing."""
    return torch.randn(*shape) * (torch.rand(shape[-1]) * 3 + 0.2)


def weighted_mse(recon: torch.Tensor, target: torch.Tensor,
                 channel_weight: torch.Tensor) -> float:
    return (channel_weight * (recon - target).pow(2)).mean().item()


def heavy_channels(in_: int, n_heavy: int = 8, weight: float = 50.0) -> torch.Tensor:
    """Channel importances with the first `n_heavy` channels dominant."""
    cw = torch.ones(in_)
    cw[:n_heavy] = weight
    return cw
