import copy
import pytest
import torch

from conftest import TinyExperts, heavy_channels, heterogeneous, weighted_mse, wrap_in_layers
from smart_quant.compensate import damped_inverse
from smart_quant.encode import centroids_for_bits, quantize_experts, quantize_fused_experts
from smart_quant.expert_importance import ActivationImportanceProfiler, HessianProfiler


class TestCentroidsForBits:
    def test_two_bits_four_dim_is_256(self):
        assert centroids_for_bits(2.0, 4) == 256  # 2^(2*4)

    def test_more_bits_more_centroids(self):
        assert centroids_for_bits(1.5, 4) < centroids_for_bits(2.0, 4) < centroids_for_bits(2.5, 4)

    def test_clamped_to_bounds(self):
        assert centroids_for_bits(0.1, 4) == 16
        assert centroids_for_bits(9.0, 4) == 4096


class TestQuantizeFusedExperts:
    def test_writes_reconstruction_in_place(self):
        torch.manual_seed(0)
        w = torch.randn(4, 64, 32)
        orig = w.clone()
        quantize_fused_experts(w, torch.full((4,), 2.0), sub_dim=4, iters=10)
        assert not torch.equal(w, orig)                # weights changed
        assert torch.isfinite(w).all()
        assert (w - orig).norm() / orig.norm() < 1.0   # a reconstruction, not noise

    def test_more_bits_lower_error(self):
        torch.manual_seed(1)
        base = torch.randn(2, 128, 32)
        low, high = base.clone(), base.clone()
        quantize_fused_experts(low, torch.full((2,), 1.5), sub_dim=4, iters=15)
        quantize_fused_experts(high, torch.full((2,), 3.0), sub_dim=4, iters=15)
        assert (high - base).norm() < (low - base).norm()

    def test_order2_matches_footprint_and_reconstructs(self):
        torch.manual_seed(2)
        w = torch.randn(2, 1024, 512)
        base = w.clone()
        bits1, n1 = quantize_fused_experts(w.clone(), torch.full((2,), 2.6), sub_dim=4,
                                           iters=5, codebook_order=1)
        w2 = base.clone()
        bits2, n2 = quantize_fused_experts(w2, torch.full((2,), 2.6), sub_dim=4,
                                           iters=5, codebook_order=2)
        assert bits2 / n2 == pytest.approx(bits1 / n1, abs=0.15)   # matched footprint
        assert (w2 - base).norm() < base.norm()                    # a reconstruction


def encode(weight: torch.Tensor, bits: float, channel_weight: torch.Tensor | None = None):
    return quantize_fused_experts(weight, torch.full((weight.shape[0],), bits), sub_dim=4,
                                  iters=15, channel_weight=channel_weight)


class TestWeightedFusedExperts:
    def test_footprint_is_unchanged_by_weighting(self):
        # the matched-comparison guarantee: weights move centroids, never bit counts
        torch.manual_seed(3)
        w = heterogeneous(2, 64, 128)
        plain = encode(w.clone(), 2.0)
        weighted = encode(w.clone(), 2.0, torch.rand(2, 128) + 0.1)
        assert plain == weighted

    @pytest.mark.parametrize("region", [slice(0, 16), slice(None)])
    def test_weighting_improves_the_channels_it_favours(self, region):
        # slice(0,16): the heavy channels reconstruct better. slice(None): the global weighted
        # objective falls — the property the fit actually optimizes.
        torch.manual_seed(1)
        base = heterogeneous(1, 64, 128)
        cw = heavy_channels(128, n_heavy=16)
        plain, weighted = base.clone(), base.clone()
        encode(plain, 1.5)
        encode(weighted, 1.5, cw.expand(1, -1))
        scale = 1.0 if region == slice(0, 16) else cw
        assert (weighted_mse(weighted[0, :, region], base[0, :, region], scale)
                < weighted_mse(plain[0, :, region], base[0, :, region], scale))

    def test_per_expert_weights_favour_each_expert_separately(self):
        # exercises the (num_experts, in_) path: expert 0 favours the low channels, expert 1 the
        # high ones, so neither can be explained by a shared codebook accident
        torch.manual_seed(4)
        base = heterogeneous(2, 64, 128)
        cw = torch.stack([heavy_channels(128, 16), heavy_channels(128, 16).flip(0)])
        plain, weighted = base.clone(), base.clone()
        encode(plain, 1.5)
        encode(weighted, 1.5, cw)
        def err(t, e, sl):
            return (t[e, :, sl] - base[e, :, sl]).pow(2).mean()

        assert err(weighted, 0, slice(0, 16)) < err(plain, 0, slice(0, 16))
        assert err(weighted, 1, slice(-16, None)) < err(plain, 1, slice(-16, None))


class TestImportanceKeyContract:
    """The profiler and the encode load the model through different wrappers; keys must bridge."""

    def test_profiler_keys_match_what_quantize_experts_looks_up(self):
        # Producer nests one level deep (AutoModel), consumer two (AutoModelForCausalLM). Keying
        # by module path would miss every tensor and silently run unweighted.
        torch.manual_seed(0)
        producer = wrap_in_layers(TinyExperts(), prefix_depth=1)
        with ActivationImportanceProfiler(producer, num_experts=4) as prof:
            producer.inner.layers[0].mlp.experts(
                torch.randn(8, 8), torch.zeros(8, 1, dtype=torch.long), torch.ones(8, 1))
        importance = prof.importance("expert")

        # Compare weighted against unweighted rather than against the original: quantization
        # happens either way, so "weights changed" would pass even with the keys never matching.
        torch.manual_seed(2)
        base = TinyExperts()
        weighted = wrap_in_layers(copy.deepcopy(base), prefix_depth=2)
        plain = wrap_in_layers(copy.deepcopy(base), prefix_depth=2)
        assert quantize_experts(weighted, avg_bits=2.0, iters=3, importance=importance)
        quantize_experts(plain, avg_bits=2.0, iters=3)
        assert not torch.equal(weighted.inner.inner.layers[0].mlp.experts.gate_up_proj,
                               plain.inner.inner.layers[0].mlp.experts.gate_up_proj), \
            "importance was looked up but never reached the fit"

    def test_unmatched_importance_raises_instead_of_degrading(self):
        # a silent fallback here would publish a null result caused by plumbing
        torch.manual_seed(1)
        consumer = wrap_in_layers(TinyExperts(), prefix_depth=1)
        with pytest.raises(KeyError, match="no key matched"):
            quantize_experts(consumer, avg_bits=2.0, iters=3,
                             importance={"model.layers.0.mlp.experts.gate_up_proj": torch.ones(8)})

    def test_unmatched_hessians_raise_even_when_importance_matches(self):
        # the shared-counter bug: valid importance masked a wrongly-keyed Hessian, so
        # compensation silently no-opped while the results row still claimed it ran
        torch.manual_seed(5)
        producer = wrap_in_layers(TinyExperts(), prefix_depth=1)
        from smart_quant.expert_importance import ActivationImportanceProfiler
        with ActivationImportanceProfiler(producer, num_experts=4) as prof:
            producer.inner.layers[0].mlp.experts(
                torch.randn(8, 8), torch.zeros(8, 1, dtype=torch.long), torch.ones(8, 1))
        good_importance = prof.importance("expert")

        consumer = wrap_in_layers(TinyExperts(), prefix_depth=1)
        with pytest.raises(KeyError, match="hessians supplied"):
            quantize_experts(consumer, avg_bits=2.0, iters=3, importance=good_importance,
                             hessians={"model.layers.0.mlp.experts": torch.eye(8)})


class TestLatticeEncode:
    def test_lattice_changes_the_result_and_reports_a_plausible_rate(self):
        # hidden=64, inter=32 -> gate_up in_=64, down_proj in_=32, both divisible by 8, and each
        # tensor holds enough sub-vectors to address a 0.75 bpw target
        torch.manual_seed(0)
        base = TinyExperts(num_experts=8, hidden=64, inter=32)
        plain = wrap_in_layers(copy.deepcopy(base), prefix_depth=1)
        latt = wrap_in_layers(copy.deepcopy(base), prefix_depth=1)
        quantize_experts(plain, avg_bits=0.75, iters=3)
        stats = quantize_experts(latt, avg_bits=0.75, iters=3, lattice=True)
        pe = plain.inner.layers[0].mlp.experts
        le = latt.inner.layers[0].mlp.experts
        assert not torch.equal(pe.gate_up_proj, le.gate_up_proj)
        assert torch.isfinite(le.gate_up_proj).all() and torch.isfinite(le.down_proj).all()
        bpw = stats[0]["quant_bits"] / stats[0]["quant_weights"]
        assert 0.5 <= bpw <= 1.5

    def test_default_path_is_untouched(self):
        torch.manual_seed(1)
        base = TinyExperts(num_experts=8, hidden=64, inter=32)
        a = wrap_in_layers(copy.deepcopy(base), prefix_depth=1)
        b = wrap_in_layers(copy.deepcopy(base), prefix_depth=1)
        quantize_experts(a, avg_bits=2.0, iters=3)
        quantize_experts(b, avg_bits=2.0, iters=3, lattice=False)
        assert torch.equal(a.inner.layers[0].mlp.experts.gate_up_proj,
                           b.inner.layers[0].mlp.experts.gate_up_proj)


class TestRealizedBpw:
    def test_accounts_exact_weights_and_near_nominal_bpw(self):
        torch.manual_seed(0)
        # experts large enough that the shared fp16 codebook amortizes toward the nominal 2 bpw
        w = torch.randn(2, 1024, 512)
        realized_bits, n_weights = quantize_fused_experts(w, torch.full((2,), 2.0), sub_dim=4, iters=5)
        assert n_weights == 2 * 1024 * 512
        assert realized_bits / n_weights == pytest.approx(2.0, abs=0.05)


class TestCompensatedEncode:
    def test_absent_hessian_is_byte_identical(self):
        torch.manual_seed(0)
        base = heterogeneous(2, 32, 16)
        a, b = base.clone(), base.clone()
        r1 = quantize_fused_experts(a, torch.full((2,), 2.0), sub_dim=4, iters=5)
        r2 = quantize_fused_experts(b, torch.full((2,), 2.0), sub_dim=4, iters=5, hinv_chol=None)
        assert r1 == r2 and torch.equal(a, b)

    def test_footprint_is_unchanged_by_compensation(self):
        # compensation moves codes, never bit counts — the matched-comparison guarantee
        torch.manual_seed(1)
        w = heterogeneous(2, 32, 16)
        u = damped_inverse(torch.eye(16) * 2)
        plain = quantize_fused_experts(w.clone(), torch.full((2,), 2.0), sub_dim=4, iters=5)
        comp = quantize_fused_experts(w.clone(), torch.full((2,), 2.0), sub_dim=4, iters=5,
                                      hinv_chol=u, rounds=2)
        assert plain == comp

    def test_only_gate_up_is_compensated(self):
        # down_proj is the control: it must come out identical with and without a Hessian.
        # The fixture must be large enough for quantization to be lossy — the default
        # TinyExperts gives gate_up (4, 8, 8), i.e. 16 sub-vectors against a k>=16 codebook,
        # so the fit is exact, the error is zero, and compensation provably cannot do anything.
        torch.manual_seed(2)
        experts = TinyExperts(num_experts=4, hidden=32, inter=16)
        model = wrap_in_layers(copy.deepcopy(experts), prefix_depth=1)
        with HessianProfiler(model) as prof:
            model.inner.layers[0].mlp.experts(
                torch.randn(256, 32), torch.zeros(256, 1, dtype=torch.long), torch.ones(256, 1))
        hess = prof.hessians()

        plain = wrap_in_layers(copy.deepcopy(experts), prefix_depth=1)
        comp = wrap_in_layers(copy.deepcopy(experts), prefix_depth=1)
        quantize_experts(plain, avg_bits=1.0, iters=3)
        quantize_experts(comp, avg_bits=1.0, iters=3, hessians=hess, rounds=2)
        pe = plain.inner.layers[0].mlp.experts
        ce = comp.inner.layers[0].mlp.experts
        assert torch.equal(pe.down_proj, ce.down_proj)          # control untouched
        assert not torch.equal(pe.gate_up_proj, ce.gate_up_proj)

    def test_non_uniform_bits_with_compensation_raises(self):
        # per-expert allocation + compensation would silently inflate the footprint ~2.4x
        torch.manual_seed(6)
        w = heterogeneous(2, 32, 16)
        with pytest.raises(ValueError, match="uniform bits_per_expert"):
            quantize_fused_experts(w, torch.tensor([2.0, 2.5]), sub_dim=4, iters=5,
                                   hinv_chol=damped_inverse(torch.eye(16)))
