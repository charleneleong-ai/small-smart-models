"""E8 vs k-means at 2.5 bpw, pooled across experts so the shell actually binds.

The previous run was degenerate at 2^20: the pool held 262,144 sub-vectors while the shell kept
1,048,576, so nearly every sub-vector got its own code — memorisation, not quantisation. Pooling
32 experts gives 8.4M sub-vectors at d=8, eight times the shell, so the restriction binds.

Rate for the lattice is log2(distinct points used) / sub_dim. A lattice needs no stored codebook —
its points are canonically enumerable — so that is the whole cost. The scale `s` is swept to find
where the distinct count lands near 2^20, i.e. 2.5 bpw.

k-means baseline is the shipped configuration: one codebook per expert, d=4, k=1024.
"""
import json
import math
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from smart_quant.codebook import assign, lloyd_kmeans

REPO, LAYER, N_EXPERTS = "Qwen/Qwen3.6-35B-A3B", 13, 32
SCALES = [0.008, 0.010, 0.0125, 0.015, 0.020, 0.026]
CHUNK = 2_000_000


def nearest_d8(x: torch.Tensor) -> torch.Tensor:
    r = torch.round(x)
    odd = (r.sum(1) % 2 != 0)
    if odd.any():
        err = x - r
        j = err.abs().argmax(dim=1)
        rows = torch.arange(x.shape[0], device=x.device)
        flip = torch.sign(err[rows, j])
        flip[flip == 0] = 1.0
        r[rows[odd], j[odd]] += flip[odd]
    return r


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    a = nearest_d8(x)
    b = nearest_d8(x - 0.5) + 0.5
    return torch.where(((x - a).pow(2).sum(1) <= (x - b).pow(2).sum(1)).unsqueeze(1), a, b)


def lattice_stats(pool: torch.Tensor, scale: float) -> tuple[float, int]:
    """(MSE, distinct points used). Coordinates are half-integers, so 2x makes them integral and
    a polynomial hash counts distinct points without a 67M-row unique."""
    sq_err, hashes = 0.0, []
    mult = torch.tensor([1, 41, 41 ** 2, 41 ** 3, 41 ** 4, 41 ** 5, 41 ** 6, 41 ** 7],
                        dtype=torch.int64, device=pool.device)
    for i in range(0, pool.shape[0], CHUNK):
        block = pool[i:i + CHUNK]
        pts = nearest_e8(block / scale)
        sq_err += float((pts * scale - block).pow(2).sum())
        hashes.append(((pts * 2).to(torch.int64) * mult).sum(1))
    distinct = int(torch.unique(torch.cat(hashes)).numel())
    return sq_err / pool.numel(), distinct


snap = snapshot_download(REPO, allow_patterns=["*.json"])
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
key = sorted(x for x in idx if f".layers.{LAYER}.mlp.experts.gate_up_proj" in x)[0]
shard = snapshot_download(REPO, allow_patterns=[idx[key]])
with safe_open(os.path.join(shard, idx[key]), framework="pt") as f:
    w = f.get_slice(key)[:N_EXPERTS].clone().detach().float().cuda()
n_e, out, in_ = w.shape
print(f"layer {LAYER}, {n_e} experts, W{tuple(w.shape)}")

# shipped baseline: per-expert codebook, d=4, k=1024
tot_err, k = 0.0, 1024
for e in range(n_e):
    pool4 = w[e].reshape(out, in_ // 4, 4).reshape(-1, 4)
    sel = torch.linspace(0, pool4.shape[0] - 1, 8192).round().long()
    c = lloyd_kmeans(pool4[sel], k, 10)[0]
    tot_err += float((c[assign(pool4, c)].reshape(out, in_) - w[e]).pow(2).sum())
km_mse = tot_err / w.numel()
km_bpw = (out * (in_ // 4) * 10 + k * 4 * 16) / (out * in_)
print(f"\n{'method':>28} {'bpw':>7} {'recon MSE':>12} {'distinct':>10}")
print(f"{'k-means d=4 k=1024 /expert':>28} {km_bpw:>7.3f} {km_mse:>12.4e} "
      f"{n_e * k:>10}")

pool8 = w.reshape(-1, 8)
print(f"{'':>28} {'':>7} {'':>12} (pool {pool8.shape[0]:,} sub-vectors)")
for s in SCALES:
    mse, distinct = lattice_stats(pool8, s)
    binds = "" if distinct < pool8.shape[0] / 4 else "  <- shell not binding"
    print(f"{'E8 s=' + format(s, '.4f'):>28} {math.log2(distinct) / 8:>7.3f} {mse:>12.4e} "
          f"{distinct:>10,}{binds}")
