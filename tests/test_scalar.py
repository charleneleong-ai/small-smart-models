import pytest
import torch

from smart_quant.scalar import scalar_quantize


def rel_l2(recon: torch.Tensor, ref: torch.Tensor) -> float:
    return float((recon - ref).norm() / ref.norm())


class TestScalarQuantize:
    def test_constant_row_reproduces_exactly(self):
        w = torch.zeros(4, 8)
        w[1] = 3.0
        assert torch.equal(scalar_quantize(w, 2), w)

    def test_high_bits_near_exact(self):
        torch.manual_seed(0)
        w = torch.randn(64, 128)
        assert rel_l2(scalar_quantize(w, 12), w) < 1e-3

    @pytest.mark.parametrize("bits", [2, 4])
    def test_error_halves_per_bit(self, bits):
        torch.manual_seed(0)
        w = torch.randn(64, 128)
        err = rel_l2(scalar_quantize(w, bits), w)
        err_next = rel_l2(scalar_quantize(w, bits + 1), w)
        assert err_next / err == pytest.approx(0.5, rel=0.25)

    def test_fractional_bits_realize_nearest_levels(self):
        w = torch.tensor([[-1.0, 1.0]])
        assert torch.equal(scalar_quantize(w, 2.0), scalar_quantize(w, 2.4))

    def test_nonpositive_bits_reject(self):
        with pytest.raises(ValueError):
            scalar_quantize(torch.randn(4, 4), 0)
