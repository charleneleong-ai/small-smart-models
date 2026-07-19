import pytest
import torch

from smart_quant.codebook import pq_bpw, pq_dequantize, pq_quantize


class TestProductQuantization:
    def test_reconstruction_improves_with_more_centroids(self):
        torch.manual_seed(0)
        w = torch.randn(256, 64)
        errs = []
        for k in (16, 64, 256):
            codes, cb = pq_quantize(w, sub_dim=4, n_centroids=k, iters=15)
            errs.append((pq_dequantize(codes, cb) - w).pow(2).mean().item())
        assert errs[0] > errs[1] > errs[2]

    def test_shapes_and_index_range(self):
        codes, cb = pq_quantize(torch.randn(128, 32), sub_dim=8, n_centroids=64)
        assert codes.shape == (128, 4) and cb.shape == (4, 64, 8)
        assert codes.max().item() < 64
        assert pq_dequantize(codes, cb).shape == (128, 32)

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError):
            pq_quantize(torch.randn(8, 30), sub_dim=4, n_centroids=16)


class TestBpw:
    def test_codebook_overhead_exceeds_nominal(self):
        # indices alone are log2(256)/4 = 2 bpw; fp16 codebooks push it higher
        assert pq_bpw(4096, 4096, sub_dim=4, n_centroids=256) > 2.0

    def test_larger_matrix_amortizes_codebook(self):
        small = pq_bpw(2048, 4096, sub_dim=4, n_centroids=256)
        large = pq_bpw(16384, 4096, sub_dim=4, n_centroids=256)
        assert large < small
