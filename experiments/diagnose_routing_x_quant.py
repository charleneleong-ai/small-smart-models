"""Item #3 — routed-input reconstruction (Arm A) + router drift under quantization (Arm B).

Two things the end-to-end rows cannot answer on their own:

Arm A. Why did the 2×2 probe predict usage allocation would HURT shared-codebook PQ (+1.0%
reconstruction) when the corrected end-to-end row showed a −2.3% ppl win? The probe scored
reconstruction on uniformly-sampled weight rows, which cannot weight an expert by how much
*routed* traffic actually touches it. This measures output error on the real routed inputs:
capture, during an fp16 forward over the Phase-2 calibration slice, the layer-L hidden rows the
router actually sent each expert, fold them into a per-expert input Gram G_e = X_e^T X_e, and
score a quantized weight as tr((W'_e - W_e) G_e (W'_e - W_e)^T) / tr(W_e G_e W_e^T).

Arm B. Does quantization change the router's top-k selections? Run the same tokens twice — fp16,
then after `quantize_experts` at `avg_bits` uniform — capture per-layer selections, and compare
per-slot agreement, frequency L1 and hot-set Jaccard against a *content-change* null (fp16 on
disjoint row halves), which is what routing would change anyway. Depth structure falls out of the
per-layer table; layer-13 per-expert correlation with Arm A asks whether quantization reroutes
traffic away from badly-quantized experts (self-healing) or toward them (compounding).

    PYTHONPATH=src python experiments/diagnose_routing_x_quant.py
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import torch
import typer
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from smart_quant.codebook import assign, lloyd_kmeans
from smart_quant.encode import centroids_for_bits, quantize_experts
from smart_quant.expert_importance import bits_from_frequency, layer_index
from smart_quant.lattice import strided_indices

REPO, SUB_DIM, ITERS = "Qwen/Qwen3.6-35B-A3B", 4, 10
ASSIGN_CHUNK, ROUTED_CAP = 4096, 32768
HOT_SET = 8

app = typer.Typer(add_completion=False)


def assign_chunked(pool: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(pool)
    for i in range(0, pool.shape[0], ASSIGN_CHUNK):
        out[i:i + ASSIGN_CHUNK] = book[assign(pool[i:i + ASSIGN_CHUNK], book)]
    return out


def rel_err(recon: torch.Tensor, ref: torch.Tensor) -> float:
    return float((recon - ref).norm() / ref.norm())


def fit_book(pool: torch.Tensor, bits: float) -> torch.Tensor:
    """Shared-codebook Lloyd fit matching the shipped encode: k = centroids_for_bits, iters=10,
    max_fit=max(4096, k*8) strided sampling over the given sub-vector pool."""
    k = centroids_for_bits(bits, SUB_DIM)
    sample = pool[strided_indices(pool.shape[0], max(4096, k * 8), pool.device)]
    return lloyd_kmeans(sample, k, ITERS)[0]


def arm_a(weights: torch.Tensor, grams: torch.Tensor, counts: torch.Tensor,
          bits: torch.Tensor) -> tuple[float, float, list[float]]:
    """Odd-row proxy rel L2 (2×2 protocol: fit even rows, eval odd) and routed-token-weighted
    output rel error from the per-expert Gram (fit all rows — eval is on inputs, so no in-sample
    risk), summed/weighted over experts."""
    proxy = 0.0
    routed_errs: list[float] = []
    for e in range(weights.shape[0]):
        w = weights[e].float()
        even = w[0::2].reshape(-1, SUB_DIM)
        odd = w[1::2].reshape(-1, SUB_DIM)
        proxy += rel_err(assign_chunked(odd, fit_book(even, float(bits[e]))), odd)
        pool = w.reshape(-1, SUB_DIM)
        recon = assign_chunked(pool, fit_book(pool, float(bits[e]))).reshape(w.shape)
        delta = recon - w
        g = grams[e].to(w.device)
        err = math.sqrt((delta @ g * delta).sum().item()
                        / max((w @ g * w).sum().item(), 1e-12))
        routed_errs.append(err)
    n = counts.float().clamp(min=1e-12)
    routed = float((n * torch.tensor(routed_errs, device=n.device)).sum() / n.sum())
    return proxy, routed, routed_errs


class SelectionProfiler:
    """Per-layer top-k selections (n_tokens, top_k) uint8, keyed by layer index — the aligned
    tensors the fp16-vs-quant comparison needs (ExpertUsageProfiler keeps only counts)."""

    def __init__(self, model: torch.nn.Module, top_k: int, num_experts: int):
        self.model = model
        self.top_k = top_k
        self.num_experts = num_experts
        self.sels: dict[int, list[torch.Tensor]] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _is_router(self, name: str, module: torch.nn.Module) -> bool:
        if name.endswith("mlp.gate"):
            return True
        return isinstance(module, torch.nn.Linear) and module.out_features == self.num_experts

    def _make_hook(self, name: str):
        layer = layer_index(name)

        def hook(_module, _inp, output):
            if layer is None:
                return
            logits = output[0] if isinstance(output, tuple) else output
            idx = logits.topk(self.top_k, dim=-1).indices.reshape(-1, self.top_k)
            self.sels.setdefault(layer, []).append(idx.detach().to("cpu", torch.uint8))
        return hook

    def __enter__(self) -> Self:
        for name, module in self.model.named_modules():
            if self._is_router(name, module):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def selections(self) -> dict[int, torch.Tensor]:
        return {k: torch.cat(v, 0) for k, v in self.sels.items()}


class RoutedInputCapture:
    """Accumulate up to `cap` routed layer-L hidden rows per expert, for the Arm-A Gram."""

    def __init__(self, model: torch.nn.Module, layer: int, n_experts: int, cap: int):
        self.layer, self.n_experts, self.cap = layer, n_experts, cap
        self.bufs: list[torch.Tensor | None] = [None] * n_experts
        self.fill = [0] * n_experts
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            if type(module).__name__.endswith("Experts"):
                self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_index(name))))

    def _make_hook(self, layer: int | None):
        def hook(_module, args: tuple[torch.Tensor, ...]) -> None:
            if layer != self.layer:
                return
            hidden, top_k_index = args[0], args[1]
            flat = hidden.reshape(-1, hidden.shape[-1])
            for e in range(self.n_experts):
                n = self.fill[e]
                if n >= self.cap:
                    continue
                rows = flat[(top_k_index == e).any(dim=-1)]
                take = min(rows.shape[0], self.cap - n)
                if take == 0:
                    continue
                if self.bufs[e] is None:
                    self.bufs[e] = torch.empty(self.cap, flat.shape[1], dtype=flat.dtype,
                                               device=flat.device)
                self.bufs[e][n:n + take] = rows[:take]
                self.fill[e] = n + take
        return hook

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def grams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(n_experts, d, d) fp32 Gram and routed-token counts, on the buffer device."""
        d = self.bufs[0].shape[1] if self.bufs[0] is not None else 0
        device = self.bufs[0].device if self.bufs[0] is not None else "cpu"
        grams = torch.zeros(self.n_experts, d, d, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_experts, dtype=torch.int64, device=device)
        for e in range(self.n_experts):
            if self.fill[e] > 0:
                xe = self.bufs[e][:self.fill[e]].float()
                grams[e] = xe.T @ xe
                counts[e] = self.fill[e]
        return grams, counts


def freq(a: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(a.reshape(-1), minlength=num_experts).float()


def freq_l1(a: torch.Tensor, b: torch.Tensor, num_experts: int) -> float:
    return float((freq(a, num_experts) / a.numel() - freq(b, num_experts) / b.numel()).abs().sum())


def hot_jaccard(a: torch.Tensor, b: torch.Tensor, num_experts: int) -> float:
    top = min(HOT_SET, num_experts)
    ha = set(freq(a, num_experts).topk(top).indices.tolist())
    hb = set(freq(b, num_experts).topk(top).indices.tolist())
    return len(ha & hb) / max(len(ha | hb), 1)


def slot_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a == b).float().mean())


@app.command()
def main(
    model: str = typer.Option(REPO),
    freq_path: Path = typer.Option(Path("experiments/bits-per-brain/expert_freq.pt")),
    avg_bits: float = typer.Option(2.0),
    lo: float = typer.Option(1.5),
    hi: float = typer.Option(3.0),
    layer: int = typer.Option(13),
    n_experts: int = typer.Option(32),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    smoke: bool = typer.Option(False, help="2 rows, no quantization — hook/capture smoke only."),
) -> None:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModel.from_pretrained(model, torch_dtype="auto", device_map="cuda").eval()
    text_cfg = lm.config.get_text_config()
    top_k, num_experts = text_cfg.num_experts_per_tok, text_cfg.num_experts
    n = 2 if smoke else calib_rows
    rows = load_dataset("allenai/c4", "en", split="train", streaming=True)

    cached: list[torch.Tensor] = []
    with SelectionProfiler(lm, top_k, num_experts) as sel, \
            RoutedInputCapture(lm, layer, n_experts, ROUTED_CAP) as rout:
        for sample, _ in zip(rows, range(n)):  # single pass over the streaming dataset
            ids = tok(sample["text"], return_tensors="pt", truncation=True,
                      max_length=seq_len).input_ids.to("cuda")
            cached.append(ids)
            with torch.no_grad():
                lm(ids)
    sel_fp16 = sel.selections()
    grams, counts = rout.grams()
    print(f"fp16 pass over {n} rows: selections captured for "
          f"{len(sel_fp16)} layers · routed tokens per expert (layer {layer}): "
          f"[{int(counts.min())}..{int(counts.max())}]\n", flush=True)

    target = next(m for name, m in lm.named_modules()
                  if type(m).__name__.endswith("Experts") and layer_index(name) == layer)
    weights = target.gate_up_proj.detach()[:n_experts].float().cuda()

    freqs = torch.load(freq_path, weights_only=True)
    fkey = next(k for k in freqs if f"layers.{layer}." in k)
    usage = freqs[fkey][:n_experts].to(weights.device)
    alloc = bits_from_frequency(usage, avg_bits, lo=lo, hi=hi)
    uniform = torch.full((n_experts,), float(avg_bits), device=weights.device)

    print(f"arm A · layer {layer} · first {n_experts} experts · d={SUB_DIM} · avg {avg_bits} bpw · "
          f"span [{lo}, {hi}] · alloc storage mean = {alloc.mean():.2f}\n", flush=True)
    print(f"{'bits':<18} {'odd-row relL2 sum':>16} {'routed out err':>16} "
          f"{'unweighted':>12}")
    results = {}
    errs_uniform: list[float] = []
    for label, bits in (("uniform 2.0", uniform), ("usage-alloc", alloc)):
        proxy, routed, errs = arm_a(weights, grams, counts, bits)
        results[label] = (proxy, routed)
        if label == "uniform 2.0":
            errs_uniform = errs
        print(f"{label:<18} {proxy:>16.4f} {routed:>16.4f} "
              f"{math.fsum(errs) / len(errs):>12.4f}", flush=True)
    for name in ("odd-row relL2 sum", "routed out err"):
        idx = 0 if name == "odd-row relL2 sum" else 1
        u, a = results["uniform 2.0"][idx], results["usage-alloc"][idx]
        print(f"Δ {name}: {(a - u) / u * 100:+.2f}%", flush=True)

    if smoke:
        return

    # --- Arm B: quantize uniform, replay the same tokens, compare selections. ---
    with torch.no_grad():
        stats = quantize_experts(lm, avg_bits=avg_bits)
    del grams, counts
    torch.cuda.empty_cache()
    expert_bits = sum(s["quant_bits"] for s in stats)
    expert_weights = sum(s["quant_weights"] for s in stats)
    print(f"\nquantized {len(stats)} MoE layers at uniform {avg_bits} bpw → "
          f"realized {expert_bits / expert_weights:.3f} expert bpw\n", flush=True)

    with SelectionProfiler(lm, top_k, num_experts) as sel:
        for ids in cached:
            with torch.no_grad():
                lm(ids)
    sel_quant = sel.selections()

    print("arm B · fp16 vs quant top-k selections on identical tokens · "
          "content null = fp16 halves\n")
    print(f"{'layer':>5} {'slot agree':>10} {'top-1':>7} {'freq L1':>8} {'Jaccard':>8} "
          f"{'null L1':>8} {'null Jac':>8}")
    for L in sorted(sel_fp16):
        a, b = sel_fp16[L], sel_quant[L]
        half = a.shape[0] // 2
        null_l1 = freq_l1(a[:half], a[half:], num_experts)
        null_jac = hot_jaccard(a[:half], a[half:], num_experts)
        print(f"{L:>5} {slot_agreement(a, b):>10.4f} "
              f"{slot_agreement(a[:, :1], b[:, :1]):>7.4f} "
              f"{freq_l1(a, b, num_experts):>8.4f} {hot_jaccard(a, b, num_experts):>8.3f} "
              f"{null_l1:>8.4f} {null_jac:>8.3f}", flush=True)

    delta_p = (freq(sel_quant[layer], num_experts)[:n_experts]
               - freq(sel_fp16[layer], num_experts)[:n_experts]) / sel_fp16[layer].numel()
    errs = torch.tensor(errs_uniform)
    if errs.numel() > 2 and delta_p.std() > 0:
        r = torch.corrcoef(torch.stack([errs.cpu(), delta_p.cpu()]))[0, 1].item()
    else:
        r = float("nan")
    print(f"\nlayer-{layer}: routed err (uniform) vs per-expert Δfreq fp16→quant: "
          f"r = {r:+.3f}", flush=True)


if __name__ == "__main__":
    app()
