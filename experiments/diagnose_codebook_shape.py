"""Does a zero-storage codebook need to be *shaped*, or does it need to be *learned*?

Phase 9 measured E8 losing to k-means, and losing worse as the rate fell — the signature of a
codebook whose points are uniform where the data is Gaussian. Two fixes exist, and they cost very
different amounts to build:

  entropy coding   keep the uniform lattice, price cells by frequency — needs a variable-length decoder
  shape matching   keep fixed rate, draw the codewords from the source distribution

QTIP takes the second: its codebook is pseudorandomly generated from a Gaussian, so it is both
zero-storage and shape-matched. This measures whether that is sufficient before anything is built.

Four codebooks at identical index cost, so the only variable is where the codewords sit:

  kmeans-expert    learned, per expert, stored (what we ship)
  kmeans-global    learned, one for all experts, stored once  — isolates the adaptation axis
  gaussian-global  free (seeded PRNG), right shape, but points clump where Lloyd would spread them
  d4-global        free, optimally spread, wrong shape        — Phase 9's failure mode at d=4

The sweep over k is the decisive part. A trellis buys an enormous *effective* codebook, so what
matters is not the gap at any single k but whether the gap **shrinks as k grows**. Flat gap =>
learned placement is doing work no zero-storage scheme recovers, and the trellis is not worth
building. Shrinking gap => QTIP's design is justified on our weights.
"""
import json
import math
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from smart_quant.codebook import assign, lloyd_kmeans
from smart_quant.lattice import distinct_points, nearest_dn, strided_indices

REPO, LAYER = "Qwen/Qwen3.6-35B-A3B", 13
N_EXPERTS, SUB_DIM = 8, 4
FIT_POINTS, EVAL_POINTS = 262_144, 131_072
TUNE_POINTS = 16_384
MULTIPLIERS = (0.7, 0.85, 1.0, 1.15, 1.3)
KS = (256, 1024, 4096, 16384)
ASSIGN_CHUNK = 4096


def assign_chunked(pool: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    """Reconstruction by nearest codeword, chunked so the distance matrix stays bounded."""
    out = torch.empty_like(pool)
    for i in range(0, pool.shape[0], ASSIGN_CHUNK):
        blk = pool[i:i + ASSIGN_CHUNK]
        out[i:i + ASSIGN_CHUNK] = book[assign(blk, book)]
    return out


def rel_err(recon: torch.Tensor, ref: torch.Tensor) -> float:
    return float((recon - ref).norm() / ref.norm())


def d4_scaled(pool: torch.Tensor, k: int, iters: int = 20) -> float:
    """D4 rescaled so it lands ~k distinct points on this pool — the lattice analogue of choosing k."""
    # the probe must comfortably exceed k or it undercounts distinct points and the bisection
    # overshoots the rate — the failure that cost Phase 9 a 2.5-bpw run that realized as 2.75
    probe = pool[strided_indices(pool.shape[0], max(16 * k, 65_536), pool.device)]
    lo, hi = 1e-5, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        n = distinct_points(nearest_dn(probe / mid))
        lo, hi = (mid, hi) if n > k else (lo, mid)
    return hi


def gaussian_tuned(tune: torch.Tensor, k: int, sigma: torch.Tensor,
                   gen: torch.Generator) -> torch.Tensor:
    """A Gaussian codebook is only defined up to scale, and one fp16 multiplier is near-free, so give
    it the same courtesy `d4_scaled` gives the lattice.

    The cloud is drawn ONCE and rescaled — redrawing per multiplier would make this
    best-of-five-random-draws and flatter the method by luck. `tune` must come from the fit half."""
    base = torch.randn(k, SUB_DIM, generator=gen, device=sigma.device) * sigma
    return base * min(MULTIPLIERS, key=lambda m: rel_err(assign_chunked(tune, base * m), tune))


def main() -> None:
    snap = snapshot_download(REPO, allow_patterns=["*.json"])
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    key = sorted(x for x in idx if f".layers.{LAYER}.mlp.experts.gate_up_proj" in x)[0]
    shard = snapshot_download(REPO, allow_patterns=[idx[key]])
    with safe_open(os.path.join(shard, idx[key]), framework="pt") as f:
        w = f.get_slice(key)[:N_EXPERTS].clone().detach().float().cuda()

    # Fit and eval must be DISJOINT. Interleaved strided subsamples of one pool are not: with
    # EVAL_POINTS = FIT_POINTS/2 every eval index coincides with a fit index, which scores the two
    # k-means rows in-sample while the two free codebooks (one scale parameter each) have nothing to
    # memorise. That bias grows with k — 16 fit points per centroid at k=16384 — so it lands on the
    # trend this experiment exists to read. Split the pool by parity first, subsample inside a half.
    halves = [(p[0::2], p[1::2]) for p in (w[e].reshape(-1, SUB_DIM) for e in range(N_EXPERTS))]
    fits = [h[strided_indices(h.shape[0], FIT_POINTS, h.device)] for h, _ in halves]
    evals = [h[strided_indices(h.shape[0], EVAL_POINTS, h.device)] for _, h in halves]
    shared_fit = torch.cat(
        [h[strided_indices(h.shape[0], FIT_POINTS // N_EXPERTS, h.device)] for h, _ in halves])
    tune = shared_fit[strided_indices(shared_fit.shape[0], TUNE_POINTS, w.device)]
    gen = torch.Generator(device=w.device).manual_seed(0)
    sigma = shared_fit.std()

    print(f"layer {LAYER} · {N_EXPERTS} experts · d={SUB_DIM} · eval {EVAL_POINTS}/expert\n")
    print(f"{'k':>7} {'bpw':>6} {'kmeans-expert':>14} {'kmeans-global':>14} "
          f"{'gaussian-global':>16} {'d4-global':>11} {'gauss gap':>10}")

    for k in KS:
        bpw = math.log2(k) / SUB_DIM
        experts = [lloyd_kmeans(ft, k, 10)[0] for ft in fits]
        glob = lloyd_kmeans(shared_fit, k, 10)[0]
        gauss = gaussian_tuned(tune, k, sigma, gen)
        s = d4_scaled(shared_fit, k)

        rows = {
            "kmeans-expert": sum(rel_err(assign_chunked(ev, bk), ev)
                                 for ev, bk in zip(evals, experts)) / N_EXPERTS,
            "kmeans-global": sum(rel_err(assign_chunked(ev, glob), ev) for ev in evals) / N_EXPERTS,
            "gaussian-global": sum(rel_err(assign_chunked(ev, gauss), ev) for ev in evals) / N_EXPERTS,
            "d4-global": sum(rel_err(nearest_dn(ev / s) * s, ev) for ev in evals) / N_EXPERTS,
        }
        gap = (rows["gaussian-global"] - rows["kmeans-expert"]) / rows["kmeans-expert"] * 100
        print(f"{k:>7} {bpw:>6.2f} {rows['kmeans-expert']:>14.4f} {rows['kmeans-global']:>14.4f} "
              f"{rows['gaussian-global']:>16.4f} {rows['d4-global']:>11.4f} {gap:>9.1f}%", flush=True)


if __name__ == "__main__":
    main()
