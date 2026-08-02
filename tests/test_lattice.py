import math

import pytest
import torch

from smart_quant.lattice import (
    calibrate_scale, distinct_points, nearest_e8, quantize_e8_fused)


def on_e8(p: torch.Tensor) -> bool:
    """E8 = D8 union (D8 + 1/2). The coset is a *per-row* property — a batch legitimately mixes
    both — so every row must be all-integer or all-half-odd, and every row sum must be even."""
    frac = p - p.floor()
    row_int = (frac.abs() < 1e-5).all(1)
    row_half = ((frac - 0.5).abs() < 1e-5).all(1)
    sums = p.sum(1) % 2
    even = (sums.abs() < 1e-4) | ((sums - 2).abs() < 1e-4)
    return bool((row_int | row_half).all()) and bool(even.all())


class TestNearestE8:
    def test_returns_lattice_points(self):
        torch.manual_seed(0)
        assert on_e8(nearest_e8(torch.randn(512, 8) * 3))

    def test_lattice_points_are_fixed(self):
        # a point already on the lattice must map to itself
        torch.manual_seed(1)
        pts = nearest_e8(torch.randn(256, 8) * 3)
        assert torch.allclose(nearest_e8(pts), pts)

    def test_matches_brute_force_on_a_neighbourhood(self):
        # For x in [-0.5, 0.5]^8 the nearest E8 point has every coordinate in {-1,-0.5,0,0.5,1},
        # so enumerating both cosets over {-1,0,1}^8 covers the true optimum. All eight
        # coordinates must vary — a partial enumeration would not contain the real nearest point.
        torch.manual_seed(2)
        grid = torch.cartesian_prod(*[torch.tensor([-1.0, 0.0, 1.0])] * 8)
        d8 = grid[grid.sum(1) % 2 == 0]
        book = torch.cat([d8, d8 + 0.5])
        x = torch.rand(64, 8) - 0.5
        brute = book[torch.cdist(x, book).argmin(1)]
        assert torch.allclose((nearest_e8(x) - x).pow(2).sum(1),
                              (brute - x).pow(2).sum(1), atol=1e-5)


class TestDistinctPoints:
    def test_counts_known_distinct_points(self):
        torch.manual_seed(3)
        pts = nearest_e8(torch.randn(4096, 8) * 2)
        assert distinct_points(pts) == torch.unique(pts, dim=0).shape[0]

    def test_repeats_do_not_inflate_the_count(self):
        torch.manual_seed(4)
        pts = nearest_e8(torch.randn(128, 8) * 2)
        assert distinct_points(pts.repeat(5, 1)) == distinct_points(pts)


class TestCalibrateScale:
    def test_hits_the_target_rate(self):
        # 1.5 bpw needs 2^12 points; 200k sub-vectors clears the 4x headroom comfortably. A 2.5
        # target would need >4M sub-vectors, which is a box-scale tensor, not a unit test.
        torch.manual_seed(5)
        pool = torch.randn(200_000, 8) * 0.01
        s = calibrate_scale(pool, target_bpw=1.5)
        rate = math.ceil(math.log2(distinct_points(nearest_e8(pool / s)))) / 8
        assert abs(rate - 1.5) <= 0.125          # one bit of index granularity

    def test_coarser_scale_gives_fewer_points(self):
        # monotonicity is exactly what makes the bisection valid
        torch.manual_seed(6)
        pool = torch.randn(50_000, 8) * 0.01
        counts = [distinct_points(nearest_e8(pool / s)) for s in (0.004, 0.008, 0.016)]
        assert counts[0] > counts[1] > counts[2]

    def test_refuses_a_target_the_pool_cannot_address(self):
        # without this guard the bisection bottoms out and returns a near-lossless fit at a
        # fictional rate — one code per sub-vector, which memorizes rather than quantizes
        torch.manual_seed(7)
        with pytest.raises(ValueError, match="cannot realize"):
            calibrate_scale(torch.randn(1024, 8) * 0.01, target_bpw=2.5)


class TestQuantizeE8Fused:
    def test_writes_reconstruction_and_charges_no_codebook(self):
        torch.manual_seed(8)
        w = torch.randn(8, 64, 128) * 0.01          # 8192 sub-vectors
        orig = w.clone()
        bits, n = quantize_e8_fused(w, target_bpw=1.0)
        assert n == 8 * 64 * 128
        assert not torch.equal(w, orig) and torch.isfinite(w).all()
        # index cost plus one fp16 scale, and nothing else. A stored codebook at this dimension
        # would add ~4 bpw, which is the entire reason for using a lattice.
        used = distinct_points(w.reshape(-1, 8) / w.reshape(-1, 8).abs()[
            w.reshape(-1, 8).abs() > 0].min())
        assert bits == pytest.approx(math.ceil(math.log2(used)) * (n / 8) + 16, rel=0.15)

    def test_higher_target_gives_lower_error(self):
        torch.manual_seed(9)
        base = torch.randn(8, 64, 128) * 0.01       # 8192 sub-vectors: clears 4x for 1.25 bpw
        lo, hi = base.clone(), base.clone()
        quantize_e8_fused(lo, target_bpw=0.75)
        quantize_e8_fused(hi, target_bpw=1.25)
        assert (hi - base).pow(2).mean() < (lo - base).pow(2).mean()

    def test_chunking_is_exact(self):
        # sub-vectors are independent given the tensor's single scale, so the chunk size bounds
        # memory without changing results — the fix Phase 8 needed after its OOM
        torch.manual_seed(10)
        base = torch.randn(8, 64, 128) * 0.01
        whole, chunked = base.clone(), base.clone()
        r_whole = quantize_e8_fused(whole, target_bpw=1.0, chunk=10 ** 9)
        r_chunk = quantize_e8_fused(chunked, target_bpw=1.0, chunk=777)
        assert torch.equal(whole, chunked)
        assert r_whole == r_chunk

    @pytest.mark.parametrize("shape,sub_dim", [((1, 8, 20), 8), ((1, 8, 16), 4)])
    def test_rejects_bad_geometry(self, shape, sub_dim):
        with pytest.raises(ValueError):
            quantize_e8_fused(torch.randn(*shape), target_bpw=2.5, sub_dim=sub_dim)
