import pytest
import torch

from smart_quant.compensate import damped_inverse


def correlated_inputs(n: int, dim: int, rank: int = 8, seed: int = 0) -> torch.Tensor:
    """(n, dim) activations with genuine off-diagonal structure: a low-rank factor plus noise.
    Uncorrelated inputs leave nothing for compensation to exploit, so fixtures must not use
    plain randn."""
    g = torch.Generator().manual_seed(seed)
    factor = torch.randn(dim, rank, generator=g)
    return torch.randn(n, rank, generator=g) @ factor.T + 0.3 * torch.randn(n, dim, generator=g)


class TestDampedInverse:
    def test_is_upper_triangular_cholesky_of_inverse(self):
        x = correlated_inputs(512, 16)
        h = x.T @ x / x.shape[0]
        u = damped_inverse(h, damp=0.01)
        assert torch.equal(u, torch.triu(u))
        # U^T U reconstructs the damped inverse
        damped = h.double() + 0.01 * torch.diag(h).double().mean() * torch.eye(16, dtype=torch.float64)
        assert torch.allclose(u.T @ u, torch.linalg.inv(damped), atol=1e-6)

    def test_survives_a_singular_covariance(self):
        # fewer samples than dimensions -> rank-deficient, exactly what cold experts produce
        x = correlated_inputs(8, 16)
        u = damped_inverse(x.T @ x / 8)
        assert torch.isfinite(u).all()

    def test_dead_channel_does_not_poison_the_factor(self):
        # an all-zero input channel gives a zero row/column; without handling, Cholesky fails
        x = correlated_inputs(512, 16)
        x[:, 5] = 0.0
        u = damped_inverse(x.T @ x / 512)
        assert torch.isfinite(u).all()
