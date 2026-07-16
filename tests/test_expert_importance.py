import pytest
import torch
from torch import nn

from smart_quant.expert_importance import ExpertUsageProfiler, bits_from_frequency

HIDDEN = 8
NUM_EXPERTS = 16
N_LAYERS = 3
TOP_K = 4
TOKENS = 32


class TinyMoELayer(nn.Module):
    """One router (gate) + a shared-expert gate that must NOT be counted as a router."""

    def __init__(self, hidden: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        self.shared_expert_gate(x)
        return logits


class TinyMoE(nn.Module):
    def __init__(self, n_layers: int, hidden: int, num_experts: int):
        super().__init__()
        self.layers = nn.ModuleList(TinyMoELayer(hidden, num_experts) for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            layer(x)
        return x


@pytest.fixture
def model() -> TinyMoE:
    torch.manual_seed(0)
    return TinyMoE(N_LAYERS, HIDDEN, NUM_EXPERTS)


class TestRouterSelection:
    def test_hooks_only_routers_not_shared_gate(self, model):
        with ExpertUsageProfiler(model, top_k=TOP_K, num_experts=NUM_EXPERTS) as prof:
            model(torch.rand(TOKENS, HIDDEN))
        assert set(prof.counts) == {f"layers.{i}.gate" for i in range(N_LAYERS)}

    def test_total_selections_match_tokens_times_topk(self, model):
        passes = 3
        with ExpertUsageProfiler(model, top_k=TOP_K, num_experts=NUM_EXPERTS) as prof:
            for _ in range(passes):
                model(torch.rand(TOKENS, HIDDEN))
        for hist in prof.counts.values():
            assert len(hist) == NUM_EXPERTS
            assert int(hist.sum()) == TOKENS * TOP_K * passes

    def test_skewed_gate_makes_one_expert_dominant(self, model):
        # Force expert 0's logit to dominate for positive inputs → always in top-k.
        for layer in model.layers:
            layer.gate.weight.data.zero_()
            layer.gate.weight.data[0] = 1.0
        with ExpertUsageProfiler(model, top_k=TOP_K, num_experts=NUM_EXPERTS) as prof:
            model(torch.rand(TOKENS, HIDDEN))
        hist = prof.counts["layers.0.gate"]
        assert int(hist[0]) == TOKENS
        assert int(hist.argmax()) == 0

    def test_hooks_removed_on_exit(self, model):
        with ExpertUsageProfiler(model, top_k=TOP_K, num_experts=NUM_EXPERTS) as prof:
            model(torch.rand(TOKENS, HIDDEN))
        before = {k: v.clone() for k, v in prof.counts.items()}
        model(torch.rand(TOKENS, HIDDEN))  # no hooks now
        for k, v in prof.counts.items():
            assert torch.equal(v, before[k])

    def test_frequencies_normalized(self, model):
        with ExpertUsageProfiler(model, top_k=TOP_K, num_experts=NUM_EXPERTS) as prof:
            model(torch.rand(TOKENS, HIDDEN))
        for freq in prof.frequencies().values():
            assert freq.sum() == pytest.approx(1.0, abs=1e-5)


class TestBitAllocation:
    def test_bounded_and_monotonic_in_frequency(self):
        freq = torch.rand(NUM_EXPERTS)
        freq = freq / freq.sum()
        bits = bits_from_frequency(freq, avg_bits=2.0, lo=1.5, hi=3.0)
        assert bits.min() >= 1.5 and bits.max() <= 3.0
        # hotter experts never get fewer bits
        order = freq.argsort()
        assert torch.all(torch.diff(bits[order]) >= -1e-6)

    def test_weighted_mean_hits_target_without_clamp(self):
        # Uniform usage + avg at the [lo,hi] midpoint → rescale is a no-op, no clamping.
        freq = torch.full((NUM_EXPERTS,), 1.0 / NUM_EXPERTS)
        bits = bits_from_frequency(freq, avg_bits=2.25, lo=1.5, hi=3.0)
        weighted_mean = (bits * freq).sum() / freq.sum()
        assert weighted_mean.item() == pytest.approx(2.25, abs=1e-4)
