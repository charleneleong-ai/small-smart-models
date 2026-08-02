# Phase 8 — GPTQ-style off-diagonal error compensation

**Study:** bits-per-brain · **Branch:** `feat/gptq-compensation-phase8` (off `main`, PRs #6–#9 merged)
**Date:** 2026-07-31 · **Status:** design approved, pending spec review

## Problem

Three phases have tried to beat uniform first-order PQ by allocating capacity non-uniformly, and all
three lost at matched footprint: Phase 3 (expert-level bit allocation), Phase 6a (second-order
residual codebooks), Phase 7 (activation weighting). The
[Phase-7 post-mortem](../weighting-diagnosis.md) explains why the last one failed, and closes the
whole family:

- Weighted k-means worked mechanically — top-1%-importance channels improved 63–93%.
- But `corr(E[x²], per-channel error)` is **negative in five of six tensors**. The signal points away
  from the error.
- And `E[x²]` **is** the layerwise Hessian diagonal (`∂²L/∂Δ_ij² = 2·E[x_j²]` for
  `L = E‖Wx − Ŵx‖²`). So that was not a crude proxy for curvature; it was curvature. **No better
  diagonal exists to substitute.**

Every technique tried so far reweights *what the fit optimizes*. Phase 8 changes *the algorithm*: it
is the first to use the off-diagonal of `E[xxᵀ]`, which is where GPTQ's and VPTQ's actual advantage
lives.

## The measurement that motivates it

Off-diagonal structure in the expert input covariance, measured over 17k–23k calibration tokens on
`pi-a100-80gb` (hot experts selected via the Phase-2 `expert_freq.pt`, so the covariance is 32–45×
oversampled rather than rank-deficient):

| tensor | dim | effective rank | top-10 eigenvalues | signal/noise on \|corr\| | cond |
|---|---|---|---|---|---|
| `gate_up_proj` | 2048 | **588–711 (29–35%)** | 20–23% | 3–5× | 7e3–9e3 |
| `down_proj` | 512 | 388–439 (76–86%) | 7–12% | 2.4–3.2× | 26–82 |

`gate_up_proj`'s inputs occupy roughly a third of their available dimensions. That redundancy is
exactly what error compensation feeds on: error committed on one group has correlated groups to be
absorbed by. `down_proj`'s intermediate is close to isotropic, so there is far less to push error
onto — its correlations are real but its redundancy is not.

## Scope: `gate_up_proj` only

Three reasons, and the third is decisive:

1. It carries **⅔ of expert weights** (1024×2048 against down_proj's 2048×512).
2. It is the tensor with the exploitable structure (29–35% effective rank against 76–86%).
3. Its input is the **hidden state, shared by every expert in a layer** — so one `H` per layer
   (16 MB × 40 = 640 MB) rather than per-expert (4 GB per layer, and undersampled at 2048 dims for
   cold experts). `down_proj`'s intermediate is expert-specific by construction and has no such
   shortcut.

Leaving `down_proj` uniform also makes it a built-in control.

## Footprint is unchanged

Compensation alters **which codes get chosen**, never what is stored. Reconstruction remains a plain
`codebook[codes]` lookup, so [`pq_bpw`](../../src/smart_quant/codebook.py) is untouched and a
compensated encode is footprint-identical to its uniform pair — the same exactness Phase 7 had, for
the same reason. There is no scale vector, no side information, no budget search.

## Design

### Block-GPTQ with the group as the block

Our quantization unit is a **group of `sub_dim=4` adjacent input channels assigned atomically to one
centroid**, so per-column sequential quantization inside a group is unavailable. That is block-GPTQ
with block = group:

```
for g in 0 .. groups-1:
    codes[:, g] = assign(W[:, B_g], codebook)         # atomic, all `out` rows
    E_g         = W[:, B_g] - codebook[codes[:, g]]   # (out, sub_dim)
    W[:, after] -= E_g @ inv(Hinv[B_g, B_g]) @ Hinv[B_g, after]
```

The `sub_dim × sub_dim` inverse is trivial. Every expert in a layer shares `H` and the group order,
so the 512-step loop is **batched over experts** — ~20k batched steps for the whole model rather
than 5.2M Python iterations. (The Phase-7 profiler's first draft was killed by exactly that kind of
per-item loop; the fix is the same.)

### Rounds, and the drift they address

GPTQ compensates against a *fixed* quantizer, but our codebook is fit globally on the original
weights — as compensation rewrites not-yet-quantized columns, they drift from what the codebook was
fit on. Phase 8 alternates:

```
W_fit = W_original
for r in 0 .. rounds-1:
    codebook       = fit_kmeans(W_fit)                      # tracks where compensation lands
    W_fit, codes   = compensate_pass(W_original, codebook)  # always replays from the original
```

Each round refits the codebook on the *previous* round's compensated weights, but replays the
compensation pass from `W_original` rather than compounding it — compounding would apply the same
correction repeatedly and diverge. Only `codes` and the final `codebook` are stored.

`rounds=3`. **Reconstruction error is logged per round**, which turns the "is 3 rounds needed"
question into a measurement and simultaneously exposes the drift: if error stops improving at round
2, round 3 is waste and the log says so.

### `refit-only` control — mandatory, not optional

Refitting alone changes the result: more k-means rounds on drifted weights is a different fit even
with compensation disabled. Without a control, a win is unattributable between *compensation* and
*extra fitting*. `refit25-control` runs the identical 3 rounds with `compensate=False`.

**`gptq25` must beat both `pq25-uniform` and `refit25-control`.** Beating only the baseline means
refitting did the work.

### Hessian estimation

`HessianProfiler` in [`expert_importance.py`](../../src/smart_quant/expert_importance.py), alongside
the two existing profilers and reusing their `Experts` forward-pre hook: accumulates `XᵀX` per layer
from the incoming hidden states, one `(2048, 2048)` fp32 per layer.

`damped_inverse(H, damp=0.01)` applies GPTQ's standard `H += 0.01·mean(diag H)·I` before inversion
and Cholesky. Layer 0's covariance has condition 1e5 and cold-expert estimates are worse, so damping
is load-bearing, not decorative.

**Stated approximation:** `H` is per-*layer*, but each expert sees only its routed token subset, so
this approximates each expert's true input covariance. Phase 7 found per-expert beat per-layer for
the diagonal, so it may matter here too. Per-expert `H` at 2048 dims is undersampled for cold experts
(hot experts reached ~20k tokens; cold ones a few hundred), and costs 4 GB per layer against 640 MB
for all forty. Per-layer is the first cut; the limitation is recorded rather than hidden.

### Where the code lives

- **`src/smart_quant/compensate.py`** (new) — `damped_inverse`, `compensated_quantize`. Kept out of
  [`codebook.py`](../../src/smart_quant/codebook.py) so that module stays focused on the quantizer.
- [`encode.py`](../../src/smart_quant/encode.py) — wires it when a Hessian is supplied; absent one,
  the existing path is byte-identical.
- [`cli.py`](../../src/smart_quant/cli.py) — `profile-hessian` command, `--hessian-path` and
  `--rounds` / `--no-compensate` on `encode-eval`.

## Experiment

| label | rounds | compensation | purpose |
|---|---|---|---|
| `pq25-uniform` | — | — | baseline, **6.2137**, already measured |
| `refit25-control` | 3 | off | isolates extra k-means rounds |
| `gptq25` | 3 | on | the arm under test |
| `gptq20` | 3 | on | second footprint, mirrors every prior phase |

Sequential on `pi-a100-80gb`, plus one Hessian calibration pass.

The control runs at 2.5 only. Its job is to answer "does refitting alone help?", which is a property
of the procedure rather than of the footprint — if three rounds of refitting do nothing at 2.5, they
are not going to be what wins at 2.0. If `gptq20` beats its baseline while `gptq25` does not, that
asymmetry is itself the finding and earns a `refit20-control` then.

**Dilution:** `down_proj` stays uniform, so a gain on `gate_up_proj` reaches perplexity at roughly
**⅔ strength**. A 0.01 improvement is consistent with a ~0.015 full-method effect. Recorded now so a
small win is not under-read — and so it cannot be invoked after the fact to rescue a null.

**Success:** `gptq25 < refit25-control < pq25-uniform`, or any ordering where `gptq25` beats both.
A fourth negative is still a publishable verdict, and a stronger one than the previous three: it
would mean the actual GPTQ mechanism, not merely a reweighting, fails on this quantizer.

## Testing

New cases in `tests/test_compensate.py` (new area, hence a new file), plus regression coverage in
[`tests/test_encode.py`](../../tests/test_encode.py):

- **Compensation reduces `‖WX − ŴX‖²`** on synthetic correlated inputs — the objective GPTQ actually
  optimizes, and the test that says the algorithm works at all.
- **Diagonal `H` makes compensation a near-no-op.** Uncorrelated inputs leave nothing to compensate;
  this catches an implementation that "helps" for the wrong reason.
- **Block update matches a per-column GPTQ reference** at `sub_dim=1`, where block-GPTQ degenerates
  to textbook GPTQ. Block formulations are easy to get subtly wrong.
- **`rounds=1, compensate=False` is byte-identical** to today's `pq_quantize`.
- **Footprint invariance** — codes and codebook shapes unchanged; `pq_bpw` never consulted.
- **Damping survives a singular `H`** without producing NaNs.

Fixtures use heterogeneous columns and explicitly correlated inputs; the Phase-7 lesson is that an
i.i.d. fixture tests the one regime where these methods are inert.

## Out of scope

- **Per-expert Hessians** — the natural refinement if per-layer shows signal, gated on solving the
  cold-expert sampling problem.
- **`down_proj` compensation** — near-isotropic inputs, and it is the control here.
- **Activation reordering (`act order`)** — GPTQ's descending-`diag(H)` group ordering. A known
  improvement, deliberately deferred so this phase tests one thing.
- **End-loss Fisher** — the other surviving direction from the post-mortem; needs backprop, so its
  own phase.
