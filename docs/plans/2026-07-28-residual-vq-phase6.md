# Phase 6a — Residual Vector Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second-order (residual) codebook to the arch-agnostic PQ quantizer and measure whether it beats first-order PQ at matched footprint on Qwen3.6-35B-A3B experts.

**Architecture:** Three generalizations to `src/smart_quant/codebook.py` (new `residual_pq_quantize`; `pq_dequantize` and `pq_bpw` extended to M stages), keeping the single-stage path byte-identical. `encode.py` and `cli.py` gain a `codebook_order` knob that defaults to 1 (no behavior change). Two matched-footprint encodes then produce a residual curve for the quality-vs-bpw plot.

**Tech Stack:** Python 3.13, PyTorch, typer, pytest, matplotlib (viz extra).

**Spec:** [`docs/specs/2026-07-28-residual-vq-phase6-design.md`](../specs/2026-07-28-residual-vq-phase6-design.md)

## Global Constraints

- Python >=3.13; fully-typed signatures with explicit generics (`list[int]`, `tuple[A, B]`).
- The M=1 / `codebook_order=1` path must stay byte-identical to today — existing callers and tests unchanged.
- `pq_bpw` and `residual_pq_quantize` must still accept a bare `int` where a stage list is expected (Phase-5 regression safety).
- No leading underscores on module-level helpers; hoist imports to top of file.
- Tests grouped by area in the existing `tests/test_codebook.py` / `tests/test_encode.py`, classes per sub-feature; no tautological tests.
- Branch `feat/residual-vq-phase6`, stacked on `feat/equal-footprint-bpw` (PR #6). Conventional commits. No `Co-Authored-By` trailer. Run `/simplify` before the FIRST commit of the code change and fold findings into that same commit.
- Encodes run on `pi-a100-80gb` as detached PPID=1 daemons (only one fp16 model fits → sequential).

---

### Task 1: `residual_pq_quantize` — sequential residual fit

**Files:**
- Modify: `src/smart_quant/codebook.py` (add function + export)
- Test: `tests/test_codebook.py` (new `TestResidualQuantization` class)

**Interfaces:**
- Consumes: existing `pq_quantize(weight, sub_dim, n_centroids, iters, share_codebook, max_fit) -> (codes, codebooks)` and `pq_dequantize(codes, codebooks) -> Tensor`.
- Produces: `residual_pq_quantize(weight: torch.Tensor, sub_dim: int, stage_centroids: list[int], iters: int = 10, share_codebook: bool = True, max_fit: int | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor]]` — returns `(codes_list, codebooks_list)`, one entry per stage.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_codebook.py`:

```python
class TestResidualQuantization:
    def test_residual_stage_reduces_error(self):
        torch.manual_seed(0)
        w = torch.randn(256, 64)
        codes1, cb1 = residual_pq_quantize(w, sub_dim=4, stage_centroids=[64])
        codes2, cb2 = residual_pq_quantize(w, sub_dim=4, stage_centroids=[64, 64])
        err1 = (pq_dequantize(codes1[0], cb1[0]) - w).pow(2).mean().item()
        recon2 = pq_dequantize(codes2[0], cb2[0]) + pq_dequantize(codes2[1], cb2[1])
        err2 = (recon2 - w).pow(2).mean().item()
        assert err2 < err1   # second stage strictly improves the reconstruction

    def test_single_stage_matches_pq_quantize(self):
        torch.manual_seed(1)
        w = torch.randn(128, 32)
        codes_r, cb_r = residual_pq_quantize(w, sub_dim=4, stage_centroids=[128])
        codes_p, cb_p = pq_quantize(w, sub_dim=4, n_centroids=128)
        assert torch.equal(codes_r[0], codes_p)
        assert torch.equal(cb_r[0], cb_p)
```

Add `residual_pq_quantize` to the `from smart_quant.codebook import ...` line at the top of the test file.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_codebook.py::TestResidualQuantization -v`
Expected: FAIL with `ImportError: cannot import name 'residual_pq_quantize'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/smart_quant/codebook.py` after `pq_quantize`:

```python
def residual_pq_quantize(
    weight: torch.Tensor,
    sub_dim: int,
    stage_centroids: list[int],
    iters: int = 10,
    share_codebook: bool = True,
    max_fit: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Multi-stage residual product quantization: stage 0 quantizes `weight`, each later
    stage quantizes the running residual `weight - sum(recon so far)`. Returns per-stage
    (codes_list, codebooks_list); `stage_centroids=[k]` reproduces a single `pq_quantize`."""
    codes_list: list[torch.Tensor] = []
    codebooks_list: list[torch.Tensor] = []
    residual = weight
    for k in stage_centroids:
        codes, codebook = pq_quantize(residual, sub_dim, k, iters, share_codebook, max_fit)
        codes_list.append(codes)
        codebooks_list.append(codebook)
        residual = residual - pq_dequantize(codes, codebook).to(weight.dtype)
    return codes_list, codebooks_list
```

Add `"residual_pq_quantize"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_codebook.py::TestResidualQuantization -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/codebook.py tests/test_codebook.py
git commit -m "feat: residual (second-order) PQ quantizer"
```

---

### Task 2: Generalize `pq_dequantize` and `pq_bpw` to M stages

**Files:**
- Modify: `src/smart_quant/codebook.py` (`pq_dequantize`, `pq_bpw`)
- Test: `tests/test_codebook.py` (extend `TestResidualQuantization`, `TestBpw`)

**Interfaces:**
- Produces: `pq_dequantize` additionally accepts `(codes_list, codebooks_list)` and sums stages. `pq_bpw(out, in_, sub_dim, stage_centroids: int | list[int], share_codebook=True) -> float` sums per-stage bpw; a bare `int` behaves as today.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_codebook.py`:

```python
    def test_dequantize_sums_stages(self):
        torch.manual_seed(2)
        w = torch.randn(64, 32)
        codes, cbs = residual_pq_quantize(w, sub_dim=4, stage_centroids=[32, 32])
        manual = pq_dequantize(codes[0], cbs[0]) + pq_dequantize(codes[1], cbs[1])
        assert torch.equal(pq_dequantize(codes, cbs), manual)
```

Add to `TestBpw`:

```python
    def test_two_stages_sum_bpw(self):
        one = pq_bpw(2048, 512, 4, 1024)
        two = pq_bpw(2048, 512, 4, [32, 32])
        # two 5-bit stages ~= one 10-bit stage on indices; shared codebooks keep both near 2.5 bpw
        assert two == pytest.approx(pq_bpw(2048, 512, 4, 32) * 2, rel=1e-6)
        assert abs(two - one) < 0.05

    def test_int_and_list_agree(self):
        assert pq_bpw(2048, 512, 4, 256) == pq_bpw(2048, 512, 4, [256])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_codebook.py -k "sums_stages or two_stages_sum or int_and_list" -v`
Expected: FAIL — `pq_dequantize` errors on a list input; `pq_bpw` errors doing `math.log2([...])`.

- [ ] **Step 3: Write minimal implementation**

In `src/smart_quant/codebook.py`, replace the head of `pq_dequantize` to dispatch on list input:

```python
def pq_dequantize(codes: torch.Tensor | list[torch.Tensor],
                  codebooks: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct the (out, in) weight. A single (codes, codebooks) pair reconstructs one
    stage (shared 2D or per-group 3D codebook); a (codes_list, codebooks_list) pair sums the
    per-stage reconstructions of a residual quantization."""
    if isinstance(codes, list):
        return sum(pq_dequantize(c, cb) for c, cb in zip(codes, codebooks))
    out, groups = codes.shape
    if codebooks.dim() == 2:  # shared: one codebook indexes every group
        recon = codebooks[codes]
    else:  # per-group
        recon = torch.stack([codebooks[g][codes[:, g]] for g in range(groups)], dim=1)
    return recon.reshape(out, -1)
```

Replace `pq_bpw` to accept a stage list:

```python
def pq_bpw(
    out: int, in_: int, sub_dim: int, n_centroids: int | list[int], share_codebook: bool = True
) -> float:
    """Effective bits-per-weight including fp16 codebook storage, summed over residual stages.
    `n_centroids` may be a single int (one stage) or a per-stage list. Sharing a single codebook
    (vs one per group) is what drops the overhead from ~index-storage to negligible."""
    stages = n_centroids if isinstance(n_centroids, list) else [n_centroids]
    groups = in_ // sub_dim
    n_codebooks = 1 if share_codebook else groups
    total_bits = sum(
        out * groups * math.log2(k) + n_codebooks * k * sub_dim * 16 for k in stages
    )
    return total_bits / (out * in_)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_codebook.py -v`
Expected: PASS (all — new residual/bpw cases plus the pre-existing single-stage cases unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/codebook.py tests/test_codebook.py
git commit -m "feat: M-stage pq_dequantize and pq_bpw accounting"
```

---

### Task 3: Thread `codebook_order` through `encode.py`

**Files:**
- Modify: `src/smart_quant/encode.py` (`quantize_fused_experts`, `quantize_experts`)
- Test: `tests/test_encode.py` (extend `TestQuantizeFusedExperts`)

**Interfaces:**
- Consumes: `residual_pq_quantize`, `pq_dequantize`, `pq_bpw` (M-stage), existing `centroids_for_bits`.
- Produces: `quantize_fused_experts(weight, bits_per_expert, sub_dim, iters=10, codebook_order=1) -> tuple[float, int]`; `quantize_experts(..., codebook_order: int = 1)`. `order=1` splits nothing (single stage); `order>1` splits the per-expert bit budget evenly across stages via `centroids_for_bits(bits/order, sub_dim)` per stage.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_encode.py`, `TestQuantizeFusedExperts`:

```python
    def test_order2_matches_footprint_and_reconstructs(self):
        torch.manual_seed(2)
        w = torch.randn(2, 1024, 512)
        base = w.clone()
        bits1, n1 = quantize_fused_experts(w.clone(), torch.full((2,), 2.6), sub_dim=4,
                                           iters=5, codebook_order=1)
        w2 = base.clone()
        bits2, n2 = quantize_fused_experts(w2, torch.full((2,), 2.6), sub_dim=4,
                                           iters=5, codebook_order=2)
        assert n1 == n2
        assert bits2 / n2 == pytest.approx(bits1 / n1, abs=0.15)   # matched footprint
        assert (w2 - base).norm() < (base).norm()                  # a reconstruction
```

Add `codebook_order` import is not needed — it's a kwarg. Ensure `import torch` / `pytest` already present (they are).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_encode.py::TestQuantizeFusedExperts::test_order2_matches_footprint_and_reconstructs -v`
Expected: FAIL — `quantize_fused_experts() got an unexpected keyword argument 'codebook_order'`

- [ ] **Step 3: Write minimal implementation**

In `src/smart_quant/encode.py`, update the import:

```python
from smart_quant.codebook import pq_bpw, pq_dequantize, pq_quantize, residual_pq_quantize
```

Replace `quantize_fused_experts` body to branch on order:

```python
def quantize_fused_experts(
    weight: torch.Tensor, bits_per_expert: torch.Tensor, sub_dim: int, iters: int = 10,
    codebook_order: int = 1,
) -> tuple[float, int]:
    """Fake-quantize a fused (num_experts, d_in, d_out) weight in place along the expert dim,
    each expert at its own `bits_per_expert` budget. With `codebook_order > 1` the budget is
    split evenly across that many residual stages. Returns (realized_bits, n_weights): total
    stored bits (indices + shared codebooks, via `pq_bpw`) and the weight count."""
    realized_bits, n_weights = 0.0, 0
    for e in range(weight.shape[0]):
        stage_centroids = [
            centroids_for_bits(float(bits_per_expert[e]) / codebook_order, sub_dim)
            for _ in range(codebook_order)
        ]
        max_fit = max(4096, max(stage_centroids) * 8)
        codes, codebooks = residual_pq_quantize(
            weight[e], sub_dim, stage_centroids, iters=iters, max_fit=max_fit)
        weight[e] = pq_dequantize(codes, codebooks).to(weight.dtype)
        out, in_ = weight[e].shape
        realized_bits += pq_bpw(out, in_, sub_dim, stage_centroids) * out * in_
        n_weights += out * in_
    return realized_bits, n_weights
```

Thread the kwarg through `quantize_experts`: add `codebook_order: int = 1` to its signature and pass `codebook_order=codebook_order` into the `quantize_fused_experts(...)` call inside the loop.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_encode.py -v`
Expected: PASS (new order-2 test plus the pre-existing order-defaulted tests unchanged — `pq_quantize` import stays valid since it's still referenced nowhere-else? confirm: it is no longer used in encode.py, so drop it from the import to keep the linter clean).

Note: after the rewrite `pq_quantize` is unused in `encode.py`. Remove it from the import line: `from smart_quant.codebook import pq_bpw, pq_dequantize, residual_pq_quantize`.

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/encode.py tests/test_encode.py
git commit -m "feat: codebook_order knob for residual expert quantization"
```

---

### Task 4: Add `--codebook-order` to the `encode-eval` CLI

**Files:**
- Modify: `src/smart_quant/cli.py` (`encode_eval`)
- Test: none (thin typer wrapper; covered by the encode tests + the manual encode run in Task 6)

**Interfaces:**
- Consumes: `quantize_experts(..., codebook_order=...)`.
- Produces: a `codebook_order` field on each `results.jsonl` row so the plot can distinguish residual encodes.

- [ ] **Step 1: Add the option and thread it**

In `src/smart_quant/cli.py`, `encode_eval`, add after the `sub_dim` option:

```python
    codebook_order: int = typer.Option(1, help="1 = single codebook; 2 = residual second-order."),
```

Pass it into the call:

```python
    stats = quantize_experts(lm, avg_bits=avg_bits, sub_dim=sub_dim, freqs=freqs,
                             bits_lo=bits_lo, bits_hi=bits_hi, codebook_order=codebook_order)
```

Add it to the emitted row:

```python
    row = {"label": label, "model": model, "allocation": allocation, "avg_bits": avg_bits,
           "sub_dim": sub_dim, "codebook_order": codebook_order,
           "wikitext_ppl": round(ppl, 4), "moe_layers": len(stats),
           "per_expert_bits_span": span, "expert_bpw": round(expert_bpw, 3),
           "model_bpw": round(model_bpw, 3)}
```

- [ ] **Step 2: Verify the CLI parses**

Run: `.venv/bin/python -c "from smart_quant.cli import app"` then `.venv/bin/smart-quant encode-eval --help` — expect `--codebook-order` listed, default 1.

- [ ] **Step 3: Commit**

```bash
git add src/smart_quant/cli.py
git commit -m "feat: --codebook-order on encode-eval"
```

---

### Task 5: `/simplify` review + fold

**Files:** whichever the review flags (expected: `codebook.py`, `encode.py`).

- [ ] **Step 1:** Run `/simplify` (code-review skill) on the working-tree diff of Tasks 1–4 vs `feat/equal-footprint-bpw`.
- [ ] **Step 2:** Fold actionable findings into the relevant commit via `git commit --amend` (Task 1/2 code commit) or a follow-up if they span tasks. Nothing is pushed yet, so amend freely.
- [ ] **Step 3:** Re-run the full suite: `.venv/bin/pytest tests/ -q` — expect green.

---

### Task 6: Matched-footprint encodes on `pi-a100-80gb`

**Files:** appends to `experiments/bits-per-brain/results.jsonl` (gitignored, on the box).

- [ ] **Step 1:** Push the branch and pull it on the box (or rsync); confirm `.venv` resolves `residual_pq_quantize`.
- [ ] **Step 2:** Pick `avg_bits` for each point so realized `expert_bpw` lands near 2.0 and 2.6 at `codebook_order=2` (start from the Phase-5 avg_bits→expert_bpw mapping: 1.75→1.76, 2.5→2.54; residual splits evenly so target ~the same nominal). Dry-run `pq_bpw` locally to confirm the split hits the target before burning a multi-hour encode.
- [ ] **Step 3:** Launch each encode as a detached PPID=1 daemon (sequential — only one fp16 model fits):

```bash
setsid nohup .venv/bin/smart-quant encode-eval \
  --model Qwen/Qwen3.6-35B-A3B --label rvq20-uniform --avg-bits <b> --codebook-order 2 \
  </dev/null >>logs/rvq20_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
# verify PPID=1:
ps -ef | grep encode-eval | grep -v grep
```

Then repeat for `rvq26-uniform` after the first completes (`--label rvq26-uniform --avg-bits <b2>`).

- [ ] **Step 4:** Confirm two new rows in `results.jsonl` with `codebook_order: 2` and `expert_bpw` near 2.0 / 2.6.

---

### Task 7: Residual curve on the plot + Phase-6 doc section

**Files:**
- Modify: `experiments/plot_quality_vs_bpw.py` (add residual curve)
- Modify: `docs/experiments/bits-per-brain.md` (Phase-6 section)
- Regenerate: `experiments/progress/bits-per-brain/quality-vs-bpw.png` (committed)

**Interfaces:**
- Consumes: `results.jsonl` rows where `label` starts with `rvq`.

- [ ] **Step 1:** In `plot_quality_vs_bpw.py`, add a `residual_curve(rows)` sibling to `uniform_curve` that selects `r["label"].startswith("rvq")` (still sorted by `expert_bpw`), and in `render` plot it as a second `o-` line (distinct color, e.g. `#7c3aed`) labeled "residual codebook PQ (2-stage)". Guard for empty (skip the line if no rvq rows) so the plot still renders pre-encode.
- [ ] **Step 2:** Regenerate the PNG: `.venv/bin/python experiments/plot_quality_vs_bpw.py` — confirm both curves render and the file lands at `experiments/progress/bits-per-brain/quality-vs-bpw.png` (allowed by the `!experiments/progress/**/*.png` gitignore exception).
- [ ] **Step 3:** Add a `### Phase 6 — residual vector quantization` section to `docs/experiments/bits-per-brain.md`: hypothesis, the matched-budget table (first-order vs residual at ~2.0 and ~2.6 bpw), and the verdict (residual curve below / on / above first-order). Link the spec.
- [ ] **Step 4: Commit**

```bash
git add experiments/plot_quality_vs_bpw.py experiments/progress/bits-per-brain/quality-vs-bpw.png docs/experiments/bits-per-brain.md
git commit -m "docs: Phase-6 residual-VQ results + plot"
```

---

### Task 8: Open the stacked PR

- [ ] **Step 1:** Push `feat/residual-vq-phase6`.
- [ ] **Step 2:** `gh pr create --base feat/equal-footprint-bpw` (stacked on PR #6) with a single-quoted heredoc body: Summary / Test plan / Visual aid (the PNG via `?raw=true`) / Commits table / Out-of-scope (AQLM joint, `.pre-commit-config.yaml` gate, Phase 6b vptq spike). Link every symbol to source on `feat/residual-vq-phase6`.
- [ ] **Step 3:** Render-check: `gh pr view <N> --json body --jq '.body' | head -60`.
- [ ] **Step 4:** Update the `project_small_smart_models` memory with the Phase-6 verdict.

---

## Self-Review

**Spec coverage:**
- Residual quantizer → Task 1. ✓
- `pq_dequantize`/`pq_bpw` M-stage → Task 2. ✓
- `codebook_order` in encode → Task 3; in CLI → Task 4. ✓
- Matched-budget discipline → Task 3 (even split) + Task 6 (dry-run `pq_bpw` before encode). ✓
- Two encode points (~2.0/~2.6) → Task 6. ✓
- Testing (residual error decreases, M=1 identity, M-stage bpw) → Tasks 1–2. ✓
- Plot residual curve + doc section → Task 7. ✓
- Stacked PR + 6b deferred → Task 8 / out-of-scope. ✓

**Type consistency:** `residual_pq_quantize` returns `(list[Tensor], list[Tensor])` consumed by `pq_dequantize(list, list)` in Tasks 2–3; `pq_bpw` takes `int | list[int]` used with a list in Task 3 and an int in existing tests. `stage_centroids` is the consistent name across spec, Task 1, and Task 3. ✓

**Placeholder scan:** every code step carries real code; encode `avg_bits` values in Task 6 are marked `<b>`/`<b2>` deliberately (resolved by the Step-2 dry-run, not guessable offline). ✓
