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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Family:
    """One encode family's line on the chart: how to select its rows and how to draw them."""
    prefix: str
    color: str
    annot_color: str
    legend: str
    offset: tuple[int, int]


# Order matters only for legend order. `uniform` is the residual of the others, so it is matched
# by exclusion rather than by prefix — adding a family means adding a row here and nothing else.
FAMILIES = (
    Family("rvq", "#7c3aed", "#5b21b6", "residual codebook PQ (2-stage)", (6, -14)),
    Family("wpq", "#0d9488", "#115e59", "activation-weighted PQ", (-9, 5)),
    Family("gptq", "#ea580c", "#9a3412", "GPTQ error compensation", (6, 8)),
    Family("e8", "#be123c", "#881337", "E8 lattice (fixed-width)", (-9, -14)),
)

# Ablation runs share a footprint with the main arm of their family (Phase-7's alpha sweep,
# Phase-8's rounds=1), so plotting them stacks two points on one x. They belong in the results
# table, not on a curve.
DIAGNOSTIC_SUFFIXES = ("-r1", "-control")


def curve(rows: list[dict[str, Any]], prefix: str | None) -> list[tuple[float, float, str]]:
    """(bpw, ppl, label) sorted by footprint, for one encode family.

    `prefix=None` selects the first-order uniform line: every `-uniform` row not claimed by a
    prefixed family. It falls back to nominal `avg_bits` for pre-instrumentation rows that
    predate realized `expert_bpw`; every prefixed family postdates it."""
    claimed = tuple(f.prefix for f in FAMILIES)
    rows = [r for r in rows if not r["label"].endswith(DIAGNOSTIC_SUFFIXES)]
    if prefix is None:
        pts = [(r.get("expert_bpw", r.get("avg_bits")), r["wikitext_ppl"], r["label"])
               for r in rows
               if r["label"].endswith("-uniform") and not r["label"].startswith(claimed)]
    else:
        pts = [(r["expert_bpw"], r["wikitext_ppl"], r["label"])
               for r in rows if r["label"].startswith(prefix)]
    return sorted(pts, key=lambda p: p[0])


def interp_ppl(curve: list[tuple[float, float, str]], bpw: float) -> float:
    """Linearly interpolate the PQ curve's perplexity at `bpw` (curve sorted ascending)."""
    for (x0, y0, _), (x1, y1, _) in zip(curve, curve[1:]):
        if x0 <= bpw <= x1:
            return y0 + (y1 - y0) * (bpw - x0) / (x1 - x0)
    raise ValueError(f"{bpw} bpw outside the swept range [{curve[0][0]}, {curve[-1][0]}]")


def draw_curve(ax, pts: list[tuple[float, float, str]], color: str, annot_color: str,
               legend: str, offset: tuple[int, int], strip: str) -> None:
    if not pts:
        return
    xs, ys, _ = zip(*pts)
    ax.plot(xs, ys, "o-", color=color, lw=2, ms=7, label=legend, zorder=3)
    for x, y, name in pts:
        # a negative x offset means "label to the left", so anchor the text's right edge —
        # families sharing a footprint must lean opposite ways or their labels stack
        ax.annotate(name.replace(strip, ""), (x, y), textcoords="offset points",
                    xytext=offset, fontsize=8, color=annot_color,
                    ha="right" if offset[0] < 0 else "left")


def render(rows: list[dict[str, Any]], out: Path) -> dict[str, float]:
    uniform = curve(rows, None)
    fp16 = next(r["wikitext_ppl"] for r in rows if r["label"] == "fp16")
    imatrix_ppl = fp16 * (1 + IMATRIX_DEGRADATION)  # llama.cpp → transformers axis
    pq_at_imatrix = interp_ppl(uniform, IMATRIX_BPW)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    draw_curve(ax, uniform, "#2563eb", "#1e3a8a", "learned codebook PQ (uniform)",
               (6, 7), "-uniform")
    for fam in FAMILIES:
        draw_curve(ax, curve(rows, fam.prefix), fam.color, fam.annot_color, fam.legend,
                   fam.offset, "-uniform")

    ax.axhline(fp16, ls="--", color="#6b7280", lw=1.3, label=f"fp16 ceiling ({fp16:.2f})")
    ax.axvline(IMATRIX_BPW, ls=":", color="#9ca3af", lw=1.2, zorder=1)
    ax.plot(IMATRIX_BPW, imatrix_ppl, "*", color="#dc2626", ms=17,
            label=f"{IMATRIX_LABEL} (+{IMATRIX_DEGRADATION:.0%} → {imatrix_ppl:.2f})", zorder=4)
    ax.plot(IMATRIX_BPW, pq_at_imatrix, "D", color="#059669", ms=9,
            label=f"PQ @ {IMATRIX_BPW} bpw (interp. {pq_at_imatrix:.2f})", zorder=4)

    gap_pct = (imatrix_ppl - pq_at_imatrix) / fp16 * 100
    # placed above the imatrix marker: the 2.5-2.6 band is crowded with same-footprint points
    # (pq25 and both wpq25 arms all sit at 2.542), so an offset to the right overlaps them
    ax.annotate(f"matched footprint\nPQ beats imatrix\nby {gap_pct:.1f} pp degradation",
                (IMATRIX_BPW, imatrix_ppl), textcoords="offset points", xytext=(10, 26),
                fontsize=8.5, color="#065f46", ha="left")

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
