import pytest
import torch

from smart_quant.encode import centroids_for_bits, quantize_fused_experts


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


class TestRealizedBpw:
    def test_accounts_exact_weights_and_near_nominal_bpw(self):
        torch.manual_seed(0)
        # experts large enough that the shared fp16 codebook amortizes toward the nominal 2 bpw
        w = torch.randn(2, 1024, 512)
        realized_bits, n_weights = quantize_fused_experts(w, torch.full((2,), 2.0), sub_dim=4, iters=5)
        assert n_weights == 2 * 1024 * 512
        assert realized_bits / n_weights == pytest.approx(2.0, abs=0.05)
