# Research item #2 — why allocation helps GGUF but hurts our PQ (2×2 mechanism)

**Study:** bits-per-brain · **Branch:** `feat/allocation-mechanism-2x2` (off `main`)
**Date:** 2026-08-03 · **Status:** design pending review

## Problem

The 2026 landscape allocates more bits to hot experts and reports gains — BitsMoE, MC-MoE, and
imatrix-GGUF all reallocate by importance or usage. Our study keeps measuring the *opposite*:
[Phase 3](../experiments/bits-per-brain.md) found usage-allocated shared-codebook PQ loses to
uniform, monotonically in allocation strength (uniform 6.77 < gentle 1.8–2.3 (7.07) < aggressive
1.5–3.0 (7.68)), and that stands at nine-plus measurements in
[local-optimum.md](../local-optimum.md). Yet the *same* reallocation helps imatrix — so the
negative result is not "reallocation is bad", it is "reallocation is bad **for a quantizer family
that shares our PQ's error structure**".

## Hypothesis: fit-limited vs rate-limited

A quantizer's error either tracks its index rate or it doesn't:

- **Rate-limited (scalar).** A per-row scalar quantizer's scale is closed-form
  (`(max − min)/levels`). There is no global fit, so error follows the rate almost exactly
  (~6 dB/bit, measured in `test_scalar.py`). Reallocating bits between experts moves error
  directly — this is why imatrix's allocation pays.
- **Fit-limited (our PQ).** Each expert's shared codebook is Lloyd-fitted on a finite sample
  (`max_fit = max(4096, k·8)`), so realized error saturates below what the index rate promises.
  The R-D curve is shallow, and reallocated bits are spent against a flat curve: hot experts buy
  little, cold experts lose a lot. Phase 10 already saw the fit wall widen with k
  (learned-vs-drawn gap 13.4% → 21.2%); this is that wall expressed as an allocation effect.

**Claim:** the *same* usage water-fill, at the *same* mean rate, should HELP the scalar family and
HURT the learned-codebook PQ — an interaction, not a uniform effect. If allocation also hurts
scalar, the verdict flips to "the usage signal is wrong for this model regardless of quantizer".

## Design — the 2×2

Instrument: `experiments/diagnose_allocation_2x2.py` — real Qwen3.6-35B-A3B layer-13
`gate_up_proj`, first 32 experts (the diagnose convention), sub-vectors `d=4`. Fit/eval halves are
disjoint (odd vs even output rows) so the high-k cells cannot win by memorising, per Phase-10
parity discipline. Allocation signal is the **real** Phase-2 routing frequency
(`experiments/bits-per-brain/expert_freq.pt`, layer-13, renormalised over the 32-expert slice).

| | uniform bits | usage-allocated (water-fill) | Δ expected |
|---|---|---|---|
| **scalar per-row** | 2.0 | `bits_from_frequency(freq, 2.0, 1.5, 3.0)` | allocation **helps** (−) |
| **PQ shared-codebook** | k=256 | per-expert `k = 2^(4·b_e)` | allocation **hurts** (+) |

- Matched on **arithmetic** mean bpw (the storage cost): `bits_from_frequency` pins the
  *usage-weighted* mean, which drifts with routing skew — the Phase-3 water-fill realized ~1.6
  bpw while targeting 2.0, so those rows were not footprint-matched. The script re-centres the
  water-fill with a shift-and-clamp (`footprint_match`) so `mean(bits) == avg_bits`, preserving
  ordering and the `[lo, hi]` range, and prints the raw drift for the record.
- Metric: sum over experts of `rel L2` on the disjoint eval half.
- Second instrument: the **R-D curve** at uniform rate `b ∈ {1.75 … 3.0}`, reporting the slope in
  dB/bit per family. Scalar should be steeper than PQ; the gap is the fit-limited wall, stated
  independently of where the allocation lands.

## Run

```bash
# on pi-a100-80gb, after pytest -q is green
setsid nohup uv run python experiments/diagnose_allocation_2x2.py \
  </dev/null >>logs/allocation_2x2_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
```

Defaults: `avg 2.0 bpw · span [1.5, 3.0]` (the aggressive span where Phase 3's negative was
strongest). A second run at the gentle span `[1.8, 2.3]` is the sensitivity line.

## Success / failure

- **Interaction confirmed:** scalar Δ < 0, PQ Δ > 0 → the fit-limited/rate-limited mechanism,
  and a publishable claim: *non-uniform MoE allocation is only worth it for quantizers whose
  error is rate-limited; learned-codebook PQ reallocates against a flat curve.*
- **Allocation helps both:** Phase 3 was an artifact (e.g. codebook-overhead asymmetry in the
  end-to-end rows) → revisit the end-to-end encode rather than the mechanism.
- **Allocation hurts both:** the usage signal itself is wrong for this model → the field's
  importance signal does not transfer to Qwen3.6 gate weights, independent of quantizer.

## Risks

- **Proxy transfer.** A reconstruction-error interaction may not become a ppl/task interaction.
  This phase is deliberately a cheap proxy to choose the next *end-to-end* encode; if the
  interaction shows, the follow-up builds a scalar quantizer into the encode path
  (`quantize_experts`) for a real ppl row.
- **Usage-weighted ≠ storage footprint.** Measured, not assumed: the first run showed the raw
  water-fill arithmetic mean at 1.59 bpw vs the 2.0 target. The table controls for it; the Phase-3
  end-to-end rows did not, so their *negative* verdict stood at a *smaller* footprint — which
  makes the negative stronger, not weaker, but the comparison was not clean. Worth a line in the
  diagnosis.
- **32-expert slice.** First-32 of 256 renormalised is a coarser allocation than Phase 3's full
  tensor. The mechanism claim does not need the full tensor; replication does, and is the
  end-to-end follow-up's job.

## Testing

New `tests/test_scalar.py` (5 cases) — constant-row exactness, high-bit recovery, error halving
per +1 bit (the rate-limited property this experiment leans on), fractional-bit level rounding,
non-positive-bits rejection. The water-fill (`bits_from_frequency`) and PQ codebook
(`lloyd_kmeans`/`assign`) are already covered by `test_expert_importance.py` / `test_codebook.py`.
The diagnose script is a probe, not unit-tested (same convention as `diagnose_lattice.py`).

## Out of scope

- **End-to-end scalar encode** — deferred until the proxy verdict; if allocation helps scalar,
  that is the build that turns the mechanism into a perplexity result.
- **Activation-importance (E[x²]) as the allocation signal** — Phase 7 showed it anti-correlates
  with PQ error; the 2×2 uses usage frequency, the signal Phase 3 and the field actually ship.
- **Real 256-expert replication** — the end-to-end leg, not this probe.
