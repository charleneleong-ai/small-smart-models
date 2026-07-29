import pytest
import torch

from conftest import heavy_channels, heterogeneous, weighted_mse
from smart_quant.codebook import (
    lloyd_kmeans, pq_bpw, pq_dequantize, pq_quantize, residual_pq_quantize)


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

    def test_max_fit_subsample_still_reconstructs_closely(self):
        torch.manual_seed(3)
        w = torch.randn(1024, 64)
        codes_full, cb_full = pq_quantize(w, 4, 256, iters=10)
        codes_sub, cb_sub = pq_quantize(w, 4, 256, iters=10, max_fit=2048)
        assert codes_sub.shape == codes_full.shape  # every sub-vector still assigned
        err_full = (pq_dequantize(codes_full, cb_full) - w).pow(2).mean()
        err_sub = (pq_dequantize(codes_sub, cb_sub) - w).pow(2).mean()
        assert err_sub < 1.5 * err_full  # fitting on a subsample barely costs quality

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

    def test_two_stages_sum_bpw(self):
        one = pq_bpw(2048, 512, 4, 1024)
        two = pq_bpw(2048, 512, 4, [32, 32])
        # Two 5-bit stages carry the same index bits as one 10-bit stage, so the only
        # difference is fp16 codebook storage: two 32-entry books cost less than one
        # 1024-entry book, making the split strictly cheaper while both stay near 2.5 bpw.
        assert two == pytest.approx(pq_bpw(2048, 512, 4, 32) * 2, rel=1e-6)
        assert two < one
        assert abs(two - one) < 0.1

    def test_int_and_list_agree(self):
        assert pq_bpw(2048, 512, 4, 256) == pq_bpw(2048, 512, 4, [256])


class TestWeightedKMeans:
    def test_uniform_weight_agrees_with_unweighted(self):
        # an explicit all-ones vector, not None — None is the default, so comparing against it
        # would assert f(x) == f(x). This exercises the weighted branch itself.
        torch.manual_seed(1)
        x = torch.randn(512, 4)
        plain = lloyd_kmeans(x, 16, iters=8)[0]
        uniform = lloyd_kmeans(x, 16, iters=8, dim_weight=torch.ones(512, 4))[0]
        assert torch.allclose(plain, uniform, atol=1e-5)

    @pytest.mark.parametrize("weight,dims", [([2.7, 0.4, 5.1, 0.9], slice(None)),
                                             ([100.0, 1.0, 1.0, 1.0], slice(0, 1))])
    def test_lowers_the_weighted_error_it_optimizes(self, weight, dims):
        # weighted k-means searches the same centroid family as the unweighted fit but scores it
        # by the weighted objective, so it cannot do worse on that objective — globally, or on
        # the one dimension a lopsided weight favours
        torch.manual_seed(2)
        x = torch.randn(2048, 4)
        w = torch.tensor(weight).expand(2048, 4)
        cu, iu = lloyd_kmeans(x, 64, iters=15)
        cw, iw = lloyd_kmeans(x, 64, iters=15, dim_weight=w)
        assert (weighted_mse(cw[iw][:, dims], x[:, dims], w[:, dims])
                < weighted_mse(cu[iu][:, dims], x[:, dims], w[:, dims]))


class TestWeightedProductQuantization:
    def test_weighted_channels_reconstruct_better(self):
        torch.manual_seed(5)
        w = heterogeneous(256, 32)
        cw = heavy_channels(32)
        plain = pq_dequantize(*pq_quantize(w, 4, 16, iters=15))
        weighted = pq_dequantize(*pq_quantize(w, 4, 16, iters=15, channel_weight=cw))
        assert ((weighted[:, :8] - w[:, :8]).pow(2).mean()
                < (plain[:, :8] - w[:, :8]).pow(2).mean())

    def test_max_fit_keeps_weights_aligned_to_points(self):
        # Reversing the weights is the misalignment control: if the subsample paired weights
        # with the wrong points, the true and reversed vectors would be equally (un)helpful.
        torch.manual_seed(6)
        w = heterogeneous(512, 64)
        cw = torch.rand(64) * 10 + 0.1

        def fit(c):
            return pq_dequantize(*pq_quantize(w, 4, 32, iters=15, max_fit=1024, channel_weight=c))

        weighted, reversed_ = fit(cw), fit(cw.flip(0))
        plain = pq_dequantize(*pq_quantize(w, 4, 32, iters=15, max_fit=1024))
        assert weighted_mse(weighted, w, cw) < weighted_mse(plain, w, cw)
        assert weighted_mse(weighted, w, cw) < weighted_mse(reversed_, w, cw)

    def test_pergroup_path_accepts_weights(self):
        torch.manual_seed(7)
        w = heterogeneous(64, 32)
        codes, cb = pq_quantize(w, 8, 16, share_codebook=False,
                                channel_weight=torch.rand(32) + 0.1)
        assert codes.shape == (64, 4) and cb.shape == (4, 16, 8)

    def test_residual_weights_every_stage_not_just_the_first(self):
        # a weight forwarded to stage 0 only would match the 1-stage result; reaching stage 1
        # must improve on it
        torch.manual_seed(8)
        w = heterogeneous(256, 32)
        cw = heavy_channels(32)
        one = pq_dequantize(*residual_pq_quantize(w, 4, [16], channel_weight=cw))
        two = pq_dequantize(*residual_pq_quantize(w, 4, [16, 16], channel_weight=cw))
        assert weighted_mse(two, w, cw) < weighted_mse(one, w, cw)


class TestResidualQuantization:
    def test_residual_stage_reduces_error(self):
        torch.manual_seed(0)
        w = torch.randn(256, 64)
        codes1, cb1 = residual_pq_quantize(w, sub_dim=4, stage_centroids=[64])
        codes2, cb2 = residual_pq_quantize(w, sub_dim=4, stage_centroids=[64, 64])
        err1 = (pq_dequantize(codes1[0], cb1[0]) - w).pow(2).mean().item()
        recon2 = pq_dequantize(codes2[0], cb2[0]) + pq_dequantize(codes2[1], cb2[1])
        err2 = (recon2 - w).pow(2).mean().item()
        assert err2 < err1   # second stage strictly improves the reconstruction

    def test_single_stage_matches_pq_quantize(self):
        torch.manual_seed(1)
        w = torch.randn(128, 32)
        codes_r, cb_r = residual_pq_quantize(w, sub_dim=4, stage_centroids=[128])
        codes_p, cb_p = pq_quantize(w, sub_dim=4, n_centroids=128)
        assert torch.equal(codes_r[0], codes_p)
        assert torch.equal(cb_r[0], cb_p)

    def test_dequantize_sums_stages(self):
        torch.manual_seed(2)
        w = torch.randn(64, 32)
        codes, cbs = residual_pq_quantize(w, sub_dim=4, stage_centroids=[32, 32])
        manual = pq_dequantize(codes[0], cbs[0]) + pq_dequantize(codes[1], cbs[1])
        assert torch.equal(pq_dequantize(codes, cbs), manual)
