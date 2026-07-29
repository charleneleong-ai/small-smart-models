"""Quality-vs-footprint plot for the bits-per-brain study: learned-codebook PQ against the
Unsloth imatrix baseline at *matched* bits-per-weight.

Reads the uniform first-order PQ sweep rows from `results.jsonl` (each carrying a realized
`expert_bpw`), overlays the phase-4 imatrix point, and — when phase-6 `rvq*` rows are present
— draws the second-order residual-VQ line beside them for a matched-footprint comparison. The
imatrix ppl was measured in llama.cpp, so it is
placed on the transformers axis by its degradation over a near-lossless reference (Q8 6.02),
applied to the fp16 ceiling — the same delta-normalization the phase-4 table uses. A vertical
locus at the imatrix footprint shows the two methods head-to-head at equal bpw.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

# Phase-4 imatrix baseline (Unsloth UD-IQ2_M, llama.cpp wikitext-2) — see docs/experiments/bits-per-brain.md
IMATRIX_LABEL = "imatrix UD-IQ2_M"
IMATRIX_BPW = 2.6
IMATRIX_DEGRADATION = 0.078  # imatrix 6.49 is +7.8% vs Q8-llama.cpp 6.02 reference


def uniform_curve(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """(bpw, ppl, label) for each first-order uniform-PQ encode, sorted by footprint. Falls
    back to the nominal `avg_bits` for pre-instrumentation rows that predate realized
    `expert_bpw`. Residual (`rvq*`) rows also end in `-uniform`, so exclude them here — they
    are the separate second-order line drawn by `residual_curve`."""
    pts = [(r.get("expert_bpw", r.get("avg_bits")), r["wikitext_ppl"], r["label"])
           for r in rows if r["label"].endswith("-uniform") and not r["label"].startswith("rvq")]
    return sorted(pts, key=lambda p: p[0])


def residual_curve(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """(bpw, ppl, label) for each residual (second-order) VQ encode, sorted by footprint.
    Every `rvq*` row carries a realized `expert_bpw`. Empty when no residual rows exist."""
    pts = [(r["expert_bpw"], r["wikitext_ppl"], r["label"])
           for r in rows if r["label"].startswith("rvq")]
    return sorted(pts, key=lambda p: p[0])


def interp_ppl(curve: list[tuple[float, float, str]], bpw: float) -> float:
    """Linearly interpolate the PQ curve's perplexity at `bpw` (curve sorted ascending)."""
    for (x0, y0, _), (x1, y1, _) in zip(curve, curve[1:]):
        if x0 <= bpw <= x1:
            return y0 + (y1 - y0) * (bpw - x0) / (x1 - x0)
    raise ValueError(f"{bpw} bpw outside the swept range [{curve[0][0]}, {curve[-1][0]}]")


def render(rows: list[dict[str, Any]], out: Path) -> dict[str, float]:
    curve = uniform_curve(rows)
    fp16 = next(r["wikitext_ppl"] for r in rows if r["label"] == "fp16")
    imatrix_ppl = fp16 * (1 + IMATRIX_DEGRADATION)  # llama.cpp → transformers axis
    pq_at_imatrix = interp_ppl(curve, IMATRIX_BPW)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    xs, ys, _ = zip(*curve)
    ax.plot(xs, ys, "o-", color="#2563eb", lw=2, ms=7, label="learned codebook PQ (uniform)", zorder=3)
    for x, y, name in curve:
        ax.annotate(name.replace("-uniform", ""), (x, y), textcoords="offset points",
                    xytext=(6, 7), fontsize=8, color="#1e3a8a")

    residual = residual_curve(rows)
    if residual:
        rxs, rys, _ = zip(*residual)
        ax.plot(rxs, rys, "o-", color="#7c3aed", lw=2, ms=7,
                label="residual codebook PQ (2-stage)", zorder=3)
        for x, y, name in residual:
            ax.annotate(name.replace("-uniform", ""), (x, y), textcoords="offset points",
                        xytext=(6, -14), fontsize=8, color="#5b21b6")

    ax.axhline(fp16, ls="--", color="#6b7280", lw=1.3, label=f"fp16 ceiling ({fp16:.2f})")
    ax.axvline(IMATRIX_BPW, ls=":", color="#9ca3af", lw=1.2, zorder=1)
    ax.plot(IMATRIX_BPW, imatrix_ppl, "*", color="#dc2626", ms=17,
            label=f"{IMATRIX_LABEL} (+{IMATRIX_DEGRADATION:.0%} → {imatrix_ppl:.2f})", zorder=4)
    ax.plot(IMATRIX_BPW, pq_at_imatrix, "D", color="#059669", ms=9,
            label=f"PQ @ {IMATRIX_BPW} bpw (interp. {pq_at_imatrix:.2f})", zorder=4)

    gap_pct = (imatrix_ppl - pq_at_imatrix) / fp16 * 100
    ax.annotate(f"matched footprint\nPQ beats imatrix\nby {gap_pct:.1f} pp degradation",
                (IMATRIX_BPW, (imatrix_ppl + pq_at_imatrix) / 2),
                textcoords="offset points", xytext=(14, -4), fontsize=8.5, color="#065f46")

    ax.set_xlabel("expert bits-per-weight (realized)")
    ax.set_ylabel("wikitext-2 perplexity  (lower = better)")
    ax.set_title("Quality vs footprint — Qwen3.6-35B-A3B experts\nlearned codebook PQ vs imatrix at matched bpw")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return {"fp16": fp16, "imatrix_norm": imatrix_ppl, "pq_at_imatrix": pq_at_imatrix, "gap_pp": gap_pct}


def main(
    results: Path = typer.Option(Path("experiments/bits-per-brain/results.jsonl")),
    out: Path = typer.Option(Path("experiments/progress/bits-per-brain/quality-vs-bpw.png")),
) -> None:
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    summary = render(rows, out)
    print(f"wrote {out}  ·  PQ {summary['pq_at_imatrix']:.3f} vs imatrix {summary['imatrix_norm']:.3f} "
          f"@ {IMATRIX_BPW} bpw  ·  gap {summary['gap_pp']:.1f} pp")


if __name__ == "__main__":
    typer.run(main)
