# Phase 7 — Activation-Weighted First-Order PQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add activation-derived importance weighting to the first-order product quantizer and measure whether it beats uniform PQ at matched footprint.

**Architecture:** Sub-vectors run along the input dim, so a sub-vector's coordinates carry different importances. The codebook fit minimizes the per-dimension weighted objective directly — weighted assignment and weighted centroid update, expanded into three matmuls costing the same as the `cdist` they replace. Weights steer the fit only; reconstruction stays a plain codebook lookup, so nothing extra is stored and a weighted encode is footprint-identical to its unweighted pair.

**Tech Stack:** Python 3.13, PyTorch (CPU for unit tests, CUDA on the box), typer CLI, pytest, matplotlib.

**Spec:** [`docs/specs/2026-07-29-weighted-pq-phase7-design.md`](../specs/2026-07-29-weighted-pq-phase7-design.md)
**Recon:** [`docs/plans/2026-07-29-phase7-recon-findings.md`](2026-07-29-phase7-recon-findings.md) — Task 1, complete, verdict **FULL**

## Global Constraints

- Branch `feat/weighted-pq-phase7`, off `main`. Never commit to `main`; the phase lands as one PR.
- Conventional commits (`feat:`, `test:`, `docs:`). No `Co-Authored-By` trailers.
- Type hints on every new signature, params **and** return, with explicit generics: `dict[str, torch.Tensor]`, never bare `dict`.
- No leading underscores on module-level functions, classes, or constants.
- Tests go in the **existing** files by area — `tests/test_codebook.py`, `tests/test_encode.py`, `tests/test_expert_importance.py`. Do not create new test modules.
- The CI surface is torch-only (`pyproject.toml` `[dependency-groups] test`). No test may import `transformers` or `datasets`.
- Every default must preserve existing behaviour: absent weights take the current code path and Phase-5/6 encodes stay byte-identical.
- Run the full suite with `PYTHONPATH=src .venv/bin/python -m pytest -q` before each commit. Baseline is **35 passed**.

### Confirmed facts from recon — do not re-derive

- Fused weights are `(num_experts, out_features, in_features)`: `gate_up_proj (256, 1024, 2048)`, `down_proj (256, 2048, 512)`. Sub-vectors run along the **input** dim.
- `Experts.forward(hidden_states, top_k_index, top_k_weights)` — routing arrives as an argument, so a single `forward_pre_hook` suffices. No router hook, no layer pairing.
- `hidden_states` is `(tokens, hidden)` already flattened; `top_k_index` is `(tokens, top_k)`.
- Intermediate recompute: `gate, up = F.linear(x_e, gate_up_proj[e]).chunk(2, -1)`, then `act_fn(gate) * up`.
- Config: `num_experts=256`, `top_k=8`, `hidden=2048`, `moe_intermediate_size=512`.

---

### ~~Task 1: Box reconnaissance~~ — COMPLETE

Committed as `55db620`. Verdict **FULL**: both projections get real statistics.

---

### ~~Task 2: `pq_bpw` scale-vector term~~ — DROPPED

Written for the pre-scaling mechanism, which needed to store `w`. Per-dimension weighted k-means stores
nothing extra, so `pq_bpw` is untouched and this task has no purpose. Implemented, then rewound.

---

### ~~Task 3: Weighted codebook fit~~ — COMPLETE (`8fb4ed5`)

Shipped as **per-dimension weighted k-means**, not the pre-scaling this plan originally specified. See
[the spec](../specs/2026-07-29-weighted-pq-phase7-design.md) for why pre-scaling was measured and
rejected: it minimizes the weighted objective over a reconstruction family that does not contain the
baseline's, and loses to it 0/5 seeds on i.i.d. data.

What landed:
- `assign(x, centroids, dim_weight=None)` — one shared metric helper so the fit and the final
  assignment cannot diverge.
- `lloyd_kmeans(..., dim_weight)` — weighted assignment *and* centroid update.
- `pq_quantize(..., channel_weight)` / `residual_pq_quantize(..., channel_weight)` — tile per-channel
  weights to the pool; `max_fit` selection hoisted to a shared `sel` so points and weights stay aligned.
- `quantize_fused_experts(..., channel_weight)` accepting `(in_,)` or `(num_experts, in_)`;
  `quantize_experts(..., importance)` keyed by full parameter name.
- `encode.py` module docstring corrected — it claimed `(num_experts, d_in, d_out)`; the real layout is
  `(num_experts, out_features, in_features)`.

Test fixtures use **heterogeneous** columns: weighting has nothing to exploit on i.i.d. data, so an
i.i.d. fixture would assert the method does nothing. 49 passed.

---

### Task 4: Activation importance profiler and shrinkage

**Files:**
- Modify: `src/smart_quant/expert_importance.py`
- Test: `tests/test_expert_importance.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `shrink_importance(raw: torch.Tensor, counts: torch.Tensor, layer_stat: torch.Tensor, tau: float = 1000.0, alpha: float = 1.0) -> torch.Tensor`
  - `ActivationImportanceProfiler(model, num_experts: int, granularity: str = "expert")` with `__enter__`/`__exit__`, `.importance() -> dict[str, torch.Tensor]` keyed by full fused parameter name, and `.counts: dict[str, torch.Tensor]`.

**Do not import `layer_index` from `encode.py`** — `encode.py` already imports `bits_from_frequency` from this module, so that direction would be a circular import. Nothing here needs it.

- [ ] **Step 1: Write the failing shrinkage tests**

Add to `tests/test_expert_importance.py`:

```python
class TestShrinkImportance:
    def test_zero_count_expert_falls_back_to_layer_stat(self):
        raw = torch.rand(3, 8) + 0.1
        counts = torch.tensor([0.0, 5000.0, 5000.0])
        layer = torch.rand(8) + 0.1
        w = shrink_importance(raw, counts, layer, tau=1000.0)
        assert torch.allclose(w[0], layer / layer.mean(), atol=1e-6)

    def test_high_count_expert_approaches_raw(self):
        raw = torch.rand(2, 8) + 0.1
        counts = torch.tensor([1e7, 1e7])
        layer = torch.rand(8) + 0.1
        w = shrink_importance(raw, counts, layer, tau=1000.0)
        assert torch.allclose(w[0], raw[0] / raw[0].mean(), atol=1e-3)

    def test_every_row_normalized_to_mean_one(self):
        raw = (torch.rand(4, 16) + 0.1) * 1e5
        w = shrink_importance(raw, torch.full((4,), 100.0), torch.rand(16) + 0.1)
        assert torch.allclose(w.mean(dim=1), torch.ones(4), atol=1e-5)

    def test_alpha_compresses_dynamic_range(self):
        raw = torch.tensor([[1.0, 100.0, 10000.0]])
        counts = torch.tensor([1e7])
        layer = torch.ones(3)
        full = shrink_importance(raw, counts, layer, alpha=1.0)
        soft = shrink_importance(raw, counts, layer, alpha=0.5)
        assert soft.max() / soft.min() < full.max() / full.min()
```

Update the import line to:

```python
from smart_quant.expert_importance import (
    ActivationImportanceProfiler, ExpertUsageProfiler, bits_from_frequency, shrink_importance)
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_expert_importance.py::TestShrinkImportance -v`
Expected: FAIL — `ImportError: cannot import name 'shrink_importance'`

- [ ] **Step 3: Implement `shrink_importance`**

Append to `src/smart_quant/expert_importance.py`:

```python
def shrink_importance(
    raw: torch.Tensor,
    counts: torch.Tensor,
    layer_stat: torch.Tensor,
    tau: float = 1000.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Empirical-Bayes shrink per-expert E[x^2] (n_experts, d_in) toward the layer statistic
    (d_in,) by routed-token count, then normalize each expert's vector to mean 1.

    At top-8 of 256 the tail experts see few tokens, so their raw statistic is noise; `tau` is
    the pseudo-count at which raw and layer contribute equally. Normalizing to mean 1 also keeps
    sqrt(w) near unity, so the scaled-space fit stays in a sane numeric range. `alpha < 1`
    compresses the dynamic range, the lever for when unmoderated magnitudes make k-means
    degenerate."""
    n = counts.to(raw.dtype).unsqueeze(1)
    w = (n * raw + tau * layer_stat.to(raw.dtype).unsqueeze(0)) / (n + tau)
    if alpha != 1.0:
        w = w.clamp(min=0).pow(alpha)
    return w / w.mean(dim=1, keepdim=True).clamp(min=1e-12)
```

- [ ] **Step 4: Implement `ActivationImportanceProfiler`**

Append to `src/smart_quant/expert_importance.py`. Accumulation is always per-expert; the layer statistic is that marginalized over experts, so the two arms of the ablation differ *only* in granularity:

```python
class ActivationImportanceProfiler:
    """Accumulates E[x^2] of each fused expert projection's input over a calibration set.

    A single forward-pre hook on each `Experts` module suffices: transformers 5 passes the
    routing in as `forward(hidden_states, top_k_index, top_k_weights)`, so there is no router
    hook to pair with. `gate_up_proj`'s statistic is over the incoming hidden states;
    `down_proj`'s is over the intermediate, recomputed per expert exactly as the forward does.

    Only running sums are retained — never raw activations — so memory is bounded at
    n_layers x n_experts x d_in floats, held on CPU.
    """

    def __init__(self, model: nn.Module, num_experts: int, granularity: str = "expert"):
        self.model = model
        self.num_experts = num_experts
        self.granularity = granularity
        self.sumsq: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(self, key: str, expert: int, x: torch.Tensor) -> None:
        """Fold x (n_tokens, d_in) into the running sum for one expert."""
        sq = x.detach().float().pow(2).sum(0).cpu()
        if key not in self.sumsq:
            self.sumsq[key] = torch.zeros(self.num_experts, sq.shape[0])
            self.counts[key] = torch.zeros(self.num_experts)
        self.sumsq[key][expert] += sq
        self.counts[key][expert] += x.shape[0]

    def make_hook(self, name: str):
        def hook(module, args):
            hidden, top_k_index = args[0], args[1]
            flat = hidden.reshape(-1, hidden.shape[-1])
            for e in torch.unique(top_k_index).tolist():
                if e >= self.num_experts:
                    continue
                token_idx = (top_k_index == e).any(dim=-1).nonzero(as_tuple=True)[0]
                if token_idx.numel() == 0:
                    continue
                x_e = flat[token_idx]
                self.add(f"{name}.gate_up_proj", e, x_e)
                gate, up = nn.functional.linear(x_e, module.gate_up_proj[e]).chunk(2, dim=-1)
                self.add(f"{name}.down_proj", e, module.act_fn(gate) * up)
        return hook

    def __enter__(self) -> "ActivationImportanceProfiler":
        for name, module in self.model.named_modules():
            if type(module).__name__.endswith("Experts"):
                self.handles.append(module.register_forward_pre_hook(self.make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def importance(self) -> dict[str, torch.Tensor]:
        """Mean x^2 per key: (n_experts, d_in) in expert mode, (d_in,) marginalized in layer mode."""
        out: dict[str, torch.Tensor] = {}
        for key, total in self.sumsq.items():
            cnt = self.counts[key]
            if self.granularity == "layer":
                out[key] = total.sum(0) / cnt.sum().clamp(min=1.0)
            else:
                out[key] = total / cnt.clamp(min=1.0).unsqueeze(1)
        return out
```

- [ ] **Step 5: Write the profiler tests**

The existing `TinyMoE` fixture predates this and does not match the transformers-5 `Experts` shape. Add a small fixture alongside it that does — fused params, and a forward taking routing — so the profiler is exercised against the real contract:

```python
class TinyExperts(nn.Module):
    """Mirrors the transformers-5 fused Experts contract the profiler hooks."""

    def __init__(self, num_experts: int = 4, hidden: int = 8, inter: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.1)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return torch.zeros_like(hidden_states)


class TestActivationImportance:
    def test_attributes_per_routed_expert(self):
        torch.manual_seed(0)
        experts = TinyExperts()
        x = torch.randn(10, 8)
        idx = torch.tensor([[0, 1]] * 6 + [[2, 3]] * 4)
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            experts(x, idx, torch.ones(10, 2))
        counts = prof.counts["gate_up_proj"]
        assert counts.tolist() == [6.0, 6.0, 4.0, 4.0]
        assert prof.importance()["gate_up_proj"].shape == (4, 8)

    def test_down_proj_statistic_matches_hand_computed_intermediate(self):
        torch.manual_seed(1)
        experts = TinyExperts()
        x = torch.randn(5, 8)
        idx = torch.zeros(5, 1, dtype=torch.long)      # every token to expert 0
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            experts(x, idx, torch.ones(5, 1))
        gate, up = nn.functional.linear(x, experts.gate_up_proj[0]).chunk(2, dim=-1)
        expected = (experts.act_fn(gate) * up).pow(2).mean(0)
        assert torch.allclose(prof.importance()["down_proj"][0], expected, atol=1e-5)

    def test_layer_mode_marginalizes_over_experts(self):
        torch.manual_seed(2)
        experts = TinyExperts()
        x = torch.randn(6, 8)
        idx = torch.tensor([[0, 1]] * 6)
        with ActivationImportanceProfiler(experts, num_experts=4,
                                          granularity="layer") as prof:
            experts(x, idx, torch.ones(6, 2))
        stat = prof.importance()["gate_up_proj"]
        assert stat.shape == (8,)
        assert torch.allclose(stat, x.pow(2).mean(0), atol=1e-5)

    def test_hooks_removed_on_exit(self):
        experts = TinyExperts()
        with ActivationImportanceProfiler(experts, num_experts=4) as prof:
            pass
        assert prof.handles == []
```

Note the keys are bare `gate_up_proj` / `down_proj` here because `named_modules()` yields `""` for the root module, so `f"{name}.{param}"` has an empty prefix. On the real model they are fully qualified.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 51 passed (43 + 8 new cases).

- [ ] **Step 7: Commit**

```bash
git add src/smart_quant/expert_importance.py tests/test_expert_importance.py
git commit -m "feat: activation importance profiler with cold-expert shrinkage"
```

---

### Task 5: CLI wiring

**Files:**
- Modify: `src/smart_quant/cli.py` — `encode-eval` (72-128), plus a `profile-activations` command

- [ ] **Step 1: Add the `profile-activations` command**

Append before `if __name__ == "__main__":`, mirroring `profile-experts`:

```python
@app.command("profile-activations")
def profile_activations(
    model: str = typer.Option(..., help="HF repo id or local path."),
    granularity: str = typer.Option("expert", help="expert | layer."),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    tau: float = typer.Option(1000.0, help="Shrinkage pseudo-count (expert granularity)."),
    alpha: float = typer.Option(1.0, help="Dynamic-range compression, w**alpha."),
    out: Path = typer.Option(Path("experiments/expert_act_importance.pt")),
) -> None:
    """Accumulate per-input-channel E[x^2] for the fused expert projections."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    from smart_quant.expert_importance import ActivationImportanceProfiler, shrink_importance

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModel.from_pretrained(model, torch_dtype="auto", device_map="cuda").eval()
    text_cfg = lm.config.get_text_config()
    rows = load_dataset("allenai/c4", "en", split="train", streaming=True)

    with ActivationImportanceProfiler(
        lm, num_experts=text_cfg.num_experts, granularity=granularity
    ) as prof:
        for _, row in zip(range(calib_rows), rows):
            ids = tok(row["text"], return_tensors="pt", truncation=True,
                      max_length=seq_len).input_ids.to("cuda")
            with torch.no_grad():
                lm(ids)
        stats = prof.importance()
        counts = dict(prof.counts)

    if granularity == "expert":
        stats = {k: shrink_importance(v, counts[k], v.mean(dim=0), tau=tau, alpha=alpha)
                 for k, v in stats.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out)
    console.print(f"profiled {len(stats)} expert tensors ({granularity}) "
                  f"over {calib_rows} rows → {out}")
```

- [ ] **Step 2: Add the two `encode-eval` options**

Add to the `encode_eval` signature, after `allocation`:

```python
    importance_path: Path = typer.Option(None, help="Activation importance .pt from profile-activations."),
    importance_granularity: str = typer.Option("expert", help="expert | layer (recorded on the row)."),
```

Load and pass it, replacing the existing `stats = quantize_experts(...)` call:

```python
    importance = torch.load(importance_path, weights_only=True) if importance_path else None
    stats = quantize_experts(lm, avg_bits=avg_bits, sub_dim=sub_dim, freqs=freqs,
                             bits_lo=bits_lo, bits_hi=bits_hi, codebook_order=codebook_order,
                             importance=importance)
```

And record it in the row dict so `results.jsonl` is self-describing — add after `"codebook_order": codebook_order,`:

```python
           "importance": importance_granularity if importance_path else None,
```

- [ ] **Step 3: Verify the CLI still loads**

Run: `PYTHONPATH=src .venv/bin/python -c "from smart_quant.cli import app; print('ok')"`
Expected: prints `ok`. No unit test — the command bodies import `transformers`, which the CI surface excludes.

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 51 passed, unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/smart_quant/cli.py
git commit -m "feat: profile-activations command and importance options on encode-eval"
```

---

### Task 6: Plot the weighted curve

**Files:**
- Modify: `experiments/plot_quality_vs_bpw.py`

- [ ] **Step 1: Add the curve selector**

Add next to `residual_curve`. The existing selectors both miss `wpq*-{expert,layer}` labels — `uniform_curve` requires `endswith("-uniform")`, `residual_curve` requires `startswith("rvq")` — so this is purely additive:

```python
def weighted_curve(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """(bpw, ppl, label) for each activation-weighted first-order encode, sorted by footprint.
    Every `wpq*` row carries a realized `expert_bpw` identical to its unweighted pair.
    Empty when no weighted rows exist."""
    pts = [(r["expert_bpw"], r["wikitext_ppl"], r["label"])
           for r in rows if r["label"].startswith("wpq")]
    return sorted(pts)
```

- [ ] **Step 2: Draw it**

Alongside the existing `ax.plot` calls, in a colour distinct from the blue first-order and purple residual lines:

```python
    weighted = weighted_curve(rows)
    if weighted:
        ax.plot([p[0] for p in weighted], [p[1] for p in weighted], "o-", color="tab:green",
                lw=2, ms=7, label="weighted PQ (activation)", zorder=3)
```

- [ ] **Step 3: Verify it regenerates unchanged before any weighted encodes exist**

Run: `PYTHONPATH=src .venv/bin/python experiments/plot_quality_vs_bpw.py`
Expected: same summary line as before — `PQ 6.193 vs imatrix 6.379 @ 2.6 bpw · gap 3.2 pp`. With no `wpq*` rows the new curve is empty and nothing is drawn.

- [ ] **Step 4: Commit**

```bash
git add experiments/plot_quality_vs_bpw.py
git commit -m "feat: weighted-PQ curve on the quality-vs-bpw plot"
```

---

### Task 7: Calibration and encodes on the box

No TDD cycle — this is the experiment. Sequential because only one fp16 model fits in 80 GB.

- [ ] **Step 1: Sync the branch to the box**

The box is at `830919f` (pre-rebase Phase 6a). Fetch and check out this branch there:

```bash
ssh pi-a100-80gb 'cd ~/small-smart-models && git fetch origin && git checkout feat/weighted-pq-phase7 && git log --oneline -1'
```

This requires the branch to be pushed first — push before running Task 7.

- [ ] **Step 2: Write the sequential launcher**

Create `run_wpq_seq.sh` on the box (not committed — matches how `run_rvq_seq.sh` was handled in Phase 6a):

```bash
#!/usr/bin/env bash
set -euo pipefail
M=Qwen/Qwen3.6-35B-A3B

for g in expert layer; do
  uv run smart-quant profile-activations --model $M --granularity $g \
    --out "experiments/expert_act_importance_$g.pt"
done

for spec in "wpq20-expert 2.0 expert" "wpq25-expert 2.5 expert" "wpq25-layer 2.5 layer"; do
  set -- $spec
  uv run smart-quant encode-eval --model $M --label "$1" --avg-bits "$2" \
    --importance-path "experiments/expert_act_importance_$3.pt" \
    --importance-granularity "$3"
done
```

- [ ] **Step 3: Launch detached, PPID=1**

```bash
setsid nohup bash run_wpq_seq.sh </dev/null \
  >>logs/wpq_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
ps -ef | grep run_wpq_seq | grep -v grep
```

Confirm column 3 (PPID) is `1`. If it is the shell's PID the run dies with the session.

- [ ] **Step 4: Verify the footprint is identical, not merely close**

As soon as `wpq20-expert` appends:

```bash
grep -E '"label": "(pq2-uniform|wpq20-expert)"' experiments/bits-per-brain/results.jsonl \
  | python -c 'import sys,json; [print(json.loads(l)["label"], json.loads(l)["expert_bpw"]) for l in sys.stdin]'
```

Expected: **both exactly `2.0`.** Weights steer the fit and change no bit count, so any difference at all means the accounting changed — stop and diagnose before the remaining encodes rather than reporting a comparison that is no longer matched.

- [ ] **Step 5: Regenerate the plot and commit the PNG**

```bash
PYTHONPATH=src .venv/bin/python experiments/plot_quality_vs_bpw.py
git add experiments/progress/bits-per-brain/quality-vs-bpw.png
git commit -m "docs: Phase-7 weighted-PQ quality-vs-bpw plot"
```

Verify the PNG is not ignored first: `git check-ignore -v experiments/progress/bits-per-brain/quality-vs-bpw.png` should return nothing.

---

### Task 8: Write up and open the PR

**Files:**
- Modify: `docs/experiments/bits-per-brain.md`

- [ ] **Step 1: Add the Phase-7 section**

After the Phase-6 section, same shape (hypothesis, matched-footprint table, verdict, design link). Fill from `results.jsonl`:

```markdown
### Phase 7 — activation-weighted first-order PQ

| footprint | uniform PQ | weighted (per-expert) | weighted (per-layer) |
|---|---|---|---|
| ~2.0 bpw | pq2 (2.00) 6.77 | wpq20-expert (2.00) _ppl_ | — |
| ~2.5 bpw | pq25 (2.54) 6.21 | wpq25-expert (2.54) _ppl_ | wpq25-layer (2.54) _ppl_ |
```

State the verdict plainly whichever way it goes, noting that footprints match exactly since weights change no bit count. If both weighted arms lose, connect it to Phase 3 and Phase 6a: uniform first-order PQ has now resisted three separate attempts at non-uniform allocation.

- [ ] **Step 2: Run the suite one final time**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: PASS — 51 passed.

- [ ] **Step 3: Simplify pass before the PR**

Run `/simplify` on the working diff and fold findings into the commits they belong to — **before** pushing, not after.

- [ ] **Step 4: Commit, push, open the PR**

```bash
git add docs/experiments/bits-per-brain.md
git commit -m "docs: Phase-7 activation-weighted PQ results"
git push -u origin feat/weighted-pq-phase7
```

Then `gh pr create` with the single-quoted heredoc form, standard section order (Summary / Test plan / Visual aid / Commits / Out-of-scope follow-ups), grouped commit tables, and every symbol deep-linked to `../tree/feat/weighted-pq-phase7/<path>`. Render-check with `gh pr view <N> --json body --jq '.body' | head -40`.

## Out of scope

- Per-sub-vector scalar weighting — an approximation of the same objective.
- Phase 6b vptq spike — still deferred.
- AQLM joint beam-search.
- `.pre-commit-config.yaml` ruff C901/PLR0915 gate — repo-wide follow-up.
