import pytest
import torch

from smart_quant.codebook import pq_bpw, pq_dequantize, pq_quantize


class TestProductQuantization:
    def test_reconstruction_improves_with_more_centroids(self):
        torch.manual_seed(0)
        w = torch.randn(256, 64)
        errs = [(pq_dequantize(*pq_quantize(w, 4, k, iters=15)) - w).pow(2).mean().item()
                for k in (16, 64, 256)]
        assert errs[0] > errs[1] > errs[2]

    def test_shared_and_pergroup_shapes(self):
        w = torch.randn(128, 32)
        codes_s, cb_s = pq_quantize(w, sub_dim=8, n_centroids=64, share_codebook=True)
        codes_p, cb_p = pq_quantize(w, sub_dim=8, n_centroids=64, share_codebook=False)
        assert cb_s.shape == (64, 8)         # one shared codebook
        assert cb_p.shape == (4, 64, 8)      # one codebook per group
        assert codes_s.shape == codes_p.shape == (128, 4)
        assert pq_dequantize(codes_s, cb_s).shape == (128, 32)
        assert pq_dequantize(codes_p, cb_p).shape == (128, 32)

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError):
            pq_quantize(torch.randn(8, 30), sub_dim=4, n_centroids=16)


class TestBpw:
    def test_shared_codebook_hits_nominal_2bpw(self):
        # log2(256)/4 = 2 bpw from indices; a shared codebook adds negligible overhead
        assert pq_bpw(2048, 512, sub_dim=4, n_centroids=256, share_codebook=True) < 2.1

    def test_pergroup_overhead_dominates(self):
        # per-group fp16 codebooks roughly double the bpw vs a single shared codebook
        shared = pq_bpw(2048, 512, 4, 256, share_codebook=True)
        pergroup = pq_bpw(2048, 512, 4, 256, share_codebook=False)
        assert pergroup > 1.9 * shared
