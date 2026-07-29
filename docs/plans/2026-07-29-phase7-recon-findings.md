# Phase 7 recon — Qwen3.6-35B-A3B `Experts` module tree

**Date:** 2026-07-29 · **Box:** `pi-a100-80gb` · **transformers:** 5.14.1
**Task:** [Task 1 of the Phase-7 plan](2026-07-29-weighted-pq-phase7.md)

Config: `num_experts=256`, `num_experts_per_tok=8`, `hidden_size=2048`, `moe_intermediate_size=512`.
Module class is `Qwen3_5MoeExperts` at `language_model.layers.{i}.mlp.experts`.

## 1. Fused parameter names and shapes

```
param: gate_up_proj (256, 1024, 2048)
param: down_proj    (256, 2048, 512)
```

## 2. Layout is `(out_features, in_features)` — not `(d_in, d_out)`

The forward uses `nn.functional.linear`, which computes `x @ W.T` and therefore takes
`W` as `(out_features, in_features)`:

```python
gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
current_hidden_states = self.act_fn(gate) * up
current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
```

Checks out dimensionally: `current_state` is `(tokens, 2048)`, `gate_up_proj[e]` is `(1024, 2048)`,
so the product is `(tokens, 1024)` and `chunk(2, -1)` yields two `(tokens, 512)` halves — exactly
`moe_intermediate_size`. Likewise `down_proj[e]` is `(2048, 512)`, mapping `(tokens, 512)` back to
`(tokens, 2048) = hidden`.

**Consequence:** `out, in_ = weight[e].shape` in
[`quantize_fused_experts`](../../src/smart_quant/encode.py) is *correctly* named — `in_` really is
`in_features`. The module docstring in [`encode.py`](../../src/smart_quant/encode.py) claiming
`(num_experts, d_in, d_out)` is wrong and should be fixed in this phase.

**Consequence for Phase 7:** `pq_quantize` splits `in_` into `sub_dim`-wide groups, so sub-vectors
run along the **input** dim. Importance `E[x_j²]` is per input channel, so the four elements of a
sub-vector have **four different weights**. The weighting is per-dimension, not per-sample.

## 3. Routing indices are passed directly to `forward`

```
SIGNATURE: (hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor
```

`top_k_index` is `(tokens, top_k)`; `hidden_states` arrives already flattened to `(tokens, hidden)`
(the forward does `final_hidden_states.index_add_(0, token_idx, ...)`).

A `forward_pre_hook` on the `Experts` module therefore receives the routing directly. **No router
hook, no pairing, and no hook-ordering hazard** — the `block_of` pairing and the ordering guard test
in the plan are both unnecessary.

For reference, `named_modules()` does emit `mlp.gate` before `mlp.experts`, so the pairing approach
would have worked — it is simply redundant now.

## 4. Gate/up packing: clean `chunk(2, dim=-1)`

The intermediate is recomputable inside the hook:

```python
gate, up = F.linear(x_e, gate_up_proj[e]).chunk(2, dim=-1)
intermediate = act_fn(gate) * up          # (n_tokens_e, 512) -> down_proj's input
```

## Verdict: **FULL**

Both projections can carry real statistics. `down_proj` does not need the uniform fallback.

## Blocker raised

Finding 2 contradicts the mechanism in
[the design spec](../specs/2026-07-29-weighted-pq-phase7-design.md) and Tasks 2–4 of the plan, both
of which assume a per-sample scalar weight. Execution paused pending the mechanism decision.
