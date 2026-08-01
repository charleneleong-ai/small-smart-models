import math

import pytest
import torch

from smart_quant.lattice import distinct_points, nearest_e8


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
