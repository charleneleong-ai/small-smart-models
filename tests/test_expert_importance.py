import pytest
import torch
from torch import nn

from conftest import TinyExperts, wrap_in_layers
from smart_quant.expert_importance import (
    ActivationImportanceProfiler, ExpertUsageProfiler, HessianProfiler, bits_from_frequency,
    normalize_importance, shrink_importance)

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


class TestShrinkImportance:
    @pytest.mark.parametrize("counts,expect_raw", [([0.0, 5e3, 5e3], False), ([1e7, 1e7, 1e7], True)])
    def test_shrinks_toward_layer_by_token_count(self, counts, expect_raw):
        raw = torch.rand(3, 8) + 0.1
        layer = torch.rand(8) + 0.1
        w = shrink_importance(raw, torch.tensor(counts), layer, tau=1000.0)
        target = raw[0] if expect_raw else layer
        assert torch.allclose(w[0], target, atol=1e-3)

    def test_normalize_scales_every_row_to_mean_one(self):
        w = normalize_importance((torch.rand(4, 16) + 0.1) * 1e5)
        assert torch.allclose(w.mean(dim=1), torch.ones(4), atol=1e-5)

    def test_alpha_compresses_dynamic_range(self):
        raw = torch.tensor([[1.0, 100.0, 10000.0]])
        full, soft = normalize_importance(raw, alpha=1.0), normalize_importance(raw, alpha=0.5)
        assert soft.max() / soft.min() < full.max() / full.min()


class TestActivationImportance:
    def test_attributes_per_routed_expert(self):
        torch.manual_seed(0)
        experts = TinyExperts()
        idx = torch.tensor([[0, 1]] * 6 + [[2, 3]] * 4)
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            experts(torch.randn(10, 8), idx, torch.ones(10, 2))
        assert prof.counts["gate_up_proj"].tolist() == [6.0, 6.0, 4.0, 4.0]
        assert prof.importance()["gate_up_proj"].shape == (4, 8)

    def test_down_proj_statistic_matches_hand_computed_intermediate(self):
        torch.manual_seed(1)
        experts = TinyExperts()
        x = torch.randn(5, 8)
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            experts(x, torch.zeros(5, 1, dtype=torch.long), torch.ones(5, 1))
        gate, up = nn.functional.linear(x, experts.gate_up_proj[0]).chunk(2, dim=-1)
        expected = (experts.act_fn(gate) * up).pow(2).mean(0)
        assert torch.allclose(prof.importance()["down_proj"][0], expected, atol=1e-5)

    def test_layer_granularity_marginalizes_the_same_pass(self):
        # one calibration pass serves both arms; layer is the token-weighted marginal of expert
        torch.manual_seed(2)
        experts = TinyExperts()
        x = torch.randn(6, 8)
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            experts(x, torch.tensor([[0, 1]] * 6), torch.ones(6, 2))
        assert prof.importance("expert")["gate_up_proj"].shape == (4, 8)
        stat = prof.importance("layer")["gate_up_proj"]
        assert stat.shape == (8,)
        assert torch.allclose(stat, x.pow(2).mean(0), atol=1e-5)

    def test_keys_are_layer_indexed_not_module_paths(self):
        # the artifact must survive AutoModel vs AutoModelForCausalLM prefix differences
        torch.manual_seed(3)

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Module()])
                self.layers[0].mlp = nn.Module()
                self.layers[0].mlp.experts = TinyExperts()

        w = Wrapper()
        with ActivationImportanceProfiler(w, num_experts=4) as prof:
            w.layers[0].mlp.experts(torch.randn(4, 8), torch.zeros(4, 1, dtype=torch.long),
                                    torch.ones(4, 1))
        assert sorted(prof.importance()) == ["0.down_proj", "0.gate_up_proj"]

    def test_hooks_removed_on_exit(self):
        experts = TinyExperts()
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            pass
        assert prof.handles == [] and not experts._forward_pre_hooks


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

    def test_weighted_mean_holds_under_skew(self):
        # The case the old rescale-then-clamp got wrong: heavy skew forces clamping, yet
        # the usage-weighted mean must still land exactly on target.
        torch.manual_seed(1)
        freq = torch.rand(NUM_EXPERTS) ** 3
        freq = freq / freq.sum()
        bits = bits_from_frequency(freq, avg_bits=2.5, lo=1.5, hi=3.0)
        assert bits.min() >= 1.5 and bits.max() <= 3.0
        assert ((bits * freq).sum()).item() == pytest.approx(2.5, abs=1e-4)

    def test_infeasible_target_clamped_to_bound(self):
        freq = torch.full((NUM_EXPERTS,), 1.0 / NUM_EXPERTS)
        assert bits_from_frequency(freq, avg_bits=5.0).min().item() == pytest.approx(3.0)


class TestHessianProfiler:
    def test_accumulates_the_input_second_moment_per_layer(self):
        torch.manual_seed(0)
        experts = TinyExperts()
        model = wrap_in_layers(experts, prefix_depth=1)
        x = torch.randn(12, 8)
        with HessianProfiler(model) as prof:
            model.inner.layers[0].mlp.experts(
                x, torch.zeros(12, 1, dtype=torch.long), torch.ones(12, 1))
        h = prof.hessians()[0]
        assert h.shape == (8, 8)
        assert torch.allclose(h, x.T @ x / 12, atol=1e-5)

    def test_keys_are_layer_indices_not_module_paths(self):
        model = wrap_in_layers(TinyExperts(), prefix_depth=2)
        with HessianProfiler(model) as prof:
            model.inner.inner.layers[0].mlp.experts(
                torch.randn(4, 8), torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1))
        assert list(prof.hessians()) == [0]

    def test_hooks_removed_on_exit(self):
        experts = TinyExperts()
        with HessianProfiler(wrap_in_layers(experts)) as prof:
            pass
        assert prof.handles == [] and not experts._forward_pre_hooks
