import torch

from smart_quant.codebook import assign, pq_dequantize, pq_quantize
from smart_quant.compensate import damped_inverse, compensated_quantize, compensated_quantize_fused
from conftest import heterogeneous


def correlated_inputs(n: int, dim: int, rank: int = 8, seed: int = 0) -> torch.Tensor:
    """(n, dim) activations with genuine off-diagonal structure: a low-rank factor plus noise.
    Uncorrelated inputs leave nothing for compensation to exploit, so fixtures must not use
    plain randn."""
    g = torch.Generator().manual_seed(seed)
    factor = torch.randn(dim, rank, generator=g)
    return torch.randn(n, rank, generator=g) @ factor.T + 0.3 * torch.randn(n, dim, generator=g)


def layer_error(w_hat: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> float:
    """||W X^T - West X^T||^2 — the objective GPTQ actually minimizes, not plain weight MSE."""
    return float(((w_hat - w) @ x.T).pow(2).mean())


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


class TestCompensatedQuantize:
    def test_lowers_the_layerwise_error_it_optimizes(self):
        torch.manual_seed(0)
        w = heterogeneous(64, 32)
        x = correlated_inputs(2048, 32, rank=6, seed=1)
        u = damped_inverse(x.T @ x / x.shape[0])
        plain = pq_dequantize(*pq_quantize(w, 4, 16, iters=10))
        codes, cb, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1)
        assert layer_error(pq_dequantize(codes, cb), w, x) < layer_error(plain, w, x)

    def test_diagonal_hessian_makes_compensation_a_near_no_op(self):
        # uncorrelated inputs leave nothing to push error onto; a version that "helps" here
        # is helping for the wrong reason
        torch.manual_seed(1)
        w = heterogeneous(64, 32)
        u = damped_inverse(torch.eye(32) * 2.0)
        on, _, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1, compensate=True)
        off, _, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1, compensate=False)
        # a diagonal H gives a diagonal Cholesky factor, so u[block, after] is exactly zero and
        # the update is a no-op — assert equality, not "close enough"
        assert torch.equal(on, off)

    def test_matches_per_column_gptq_at_sub_dim_1(self):
        # at sub_dim=1 the block update degenerates to textbook GPTQ; block formulations are
        # easy to get subtly wrong, so pin it against an explicit per-column reference
        torch.manual_seed(2)
        w = heterogeneous(16, 8)
        x = correlated_inputs(1024, 8, rank=3, seed=3)
        u = damped_inverse(x.T @ x / x.shape[0])
        codes, cb, _ = compensated_quantize(w, 1, 8, u, iters=20, rounds=1)

        work, ref = w.float().clone(), torch.empty_like(w)
        for j in range(8):
            idx = assign(work[:, j:j + 1], cb.float())
            ref[:, j] = cb.float()[idx].squeeze(1)
            err = work[:, j] - ref[:, j]
            if j + 1 < 8:
                work[:, j + 1:] -= torch.outer(err / u[j, j], u[j, j + 1:]).float()
        assert torch.allclose(pq_dequantize(codes, cb), ref, atol=1e-5)

    def test_compensate_false_is_byte_identical_to_pq_quantize(self):
        torch.manual_seed(3)
        w = heterogeneous(64, 32)
        u = damped_inverse(torch.eye(32))
        codes, cb, _ = compensated_quantize(w, 4, 16, u, iters=10, rounds=1, compensate=False)
        base_codes, base_cb = pq_quantize(w, 4, 16, iters=10)
        assert torch.equal(codes, base_codes) and torch.equal(cb, base_cb)

    def test_reports_one_error_per_round_and_shapes_are_unchanged(self):
        torch.manual_seed(4)
        w = heterogeneous(64, 32)
        x = correlated_inputs(2048, 32, rank=6, seed=5)
        u = damped_inverse(x.T @ x / x.shape[0])
        codes, cb, errs = compensated_quantize(w, 4, 16, u, iters=10, rounds=3)
        base_codes, base_cb = pq_quantize(w, 4, 16, iters=10)
        assert len(errs) == 3
        assert codes.shape == base_codes.shape and cb.shape == base_cb.shape   # footprint


class TestBatchedCompensation:
    def test_matches_the_single_tensor_reference_expert_for_expert(self):
        torch.manual_seed(0)
        base = torch.stack([heterogeneous(32, 16) for _ in range(3)])
        x = correlated_inputs(1024, 16, rank=4, seed=7)
        u = damped_inverse(x.T @ x / x.shape[0])

        batched = base.clone()
        compensated_quantize_fused(batched, 4, 16, u, iters=10, rounds=2)
        for e in range(3):
            codes, cb, _ = compensated_quantize(base[e], 4, 16, u, iters=10, rounds=2)
            assert torch.allclose(batched[e], pq_dequantize(codes, cb), atol=1e-5)

    def test_writes_reconstruction_in_place_and_reports_per_round(self):
        torch.manual_seed(1)
        w = torch.stack([heterogeneous(32, 16) for _ in range(2)])
        orig = w.clone()
        errs = compensated_quantize_fused(w, 4, 16, damped_inverse(torch.eye(16)), rounds=3)
        assert len(errs) == 3
        assert not torch.equal(w, orig) and torch.isfinite(w).all()
