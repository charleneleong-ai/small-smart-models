"""smart-quant CLI — footprint accounting, perplexity smoke test, expert profiling."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from smart_quant.footprint import Footprint, target_bytes

app = typer.Typer(add_completion=False, help="Low-bit quant benchmarking for smart small MoEs.")
console = Console()


@app.command()
def footprint(
    params: float = typer.Option(..., help="Total parameter count, e.g. 35e9."),
    bytes_: int = typer.Option(..., "--bytes", help="Quant file size in bytes."),
) -> None:
    """Report effective bits-per-weight and size for a quant file."""
    fp = Footprint(total_params=int(params), file_bytes=bytes_)
    console.print(f"[bold]{fp.gib:.2f} GiB[/bold]  ·  [bold]{fp.bpw:.2f} bpw[/bold]  "
                  f"({fp.total_params/1e9:.1f}B params)")


@app.command()
def budget(
    params: float = typer.Option(..., help="Total parameter count, e.g. 35e9."),
    bpw: float = typer.Option(..., help="Target bits-per-weight."),
) -> None:
    """Byte budget an encode must hit to land at a target bpw."""
    b = target_bytes(int(params), bpw)
    console.print(f"target: [bold]{b:,} bytes[/bold] ({b/1024**3:.2f} GiB) at {bpw} bpw")


@app.command("eval")
def eval_model(
    model: str = typer.Option(..., help="HF repo id or local path."),
    label: str = typer.Option(..., help="Row label, e.g. fp16 / iq2_m / vptq-2bit."),
    gguf_file: str = typer.Option(None, help="GGUF filename within the repo (dequantized load)."),
    tokenizer: str = typer.Option(None, help="Tokenizer repo id; defaults to --model. Point this at "
                                             "the fp16 base when evaluating a GGUF so the token "
                                             "count matches every other row in results.jsonl."),
    dataset: str = typer.Option("Salesforce/wikitext", help="HF dataset repo id."),
    config: str = typer.Option("wikitext-2-raw-v1", help="Dataset config."),
    max_length: int = typer.Option(4096),
    stride: int = typer.Option(2048),
    out: Path = typer.Option(Path("experiments/bits-per-brain/results.jsonl")),
) -> None:
    """Sliding-window wikitext perplexity for one model; append a row to results.jsonl."""
    import json

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from smart_quant.eval import load_causal_lm, sliding_window_perplexity

    # dataset first, so a bad id fails fast rather than after the multi-minute model load
    # A GGUF repo's own tokenizer is not necessarily byte-identical to the base model's, and
    # perplexity is only comparable across rows when the token count is. Default to --model so
    # existing calls are unchanged; override to the fp16 base when scoring a GGUF.
    tok = AutoTokenizer.from_pretrained(tokenizer or model)
    text = "\n\n".join(load_dataset(dataset, config, split="test")["text"])
    load_kwargs = {"dtype": "auto", "device_map": "cuda"}
    if gguf_file:
        # GGUF is dequantized on load, so a 35B build lands at ~70 GB in fp16 — close enough to an
        # 80 GB card that "auto" should be free to spill rather than OOM. fp32 would not fit at all.
        load_kwargs |= {"gguf_file": gguf_file, "dtype": "float16", "device_map": "auto"}
    lm = load_causal_lm(model, **load_kwargs).eval()
    ppl = sliding_window_perplexity(lm, tok, text, max_length, stride, "cuda")

    row = {"label": label, "model": model, "gguf_file": gguf_file,
           "tokenizer": tokenizer or model, "wikitext_ppl": round(ppl, 4),
           "dataset": f"{dataset}:{config}"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    console.print(f"[bold]{label}[/bold]  wikitext-2 ppl = [bold]{ppl:.4f}[/bold]  ->  {out}")


@app.command("encode-eval")
def encode_eval(
    model: str = typer.Option(..., help="HF repo id or local path (fp16)."),
    label: str = typer.Option(..., help="Row label, e.g. pq2-uniform / pq2-expert."),
    avg_bits: float = typer.Option(2.0, help="Target average bits/weight for the experts."),
    sub_dim: int = typer.Option(4),
    codebook_order: int = typer.Option(1, help="1 = single codebook; 2 = residual second-order."),
    allocation: str = typer.Option("uniform", help="uniform | expert (usage-driven)."),
    bits_lo: float = typer.Option(1.5, help="Min per-expert bits (expert allocation)."),
    bits_hi: float = typer.Option(3.0, help="Max per-expert bits (expert allocation)."),
    freqs_path: Path = typer.Option(Path("experiments/bits-per-brain/expert_freq.pt")),
    lattice: bool = typer.Option(
        False, help="Quantize to the E8 lattice instead of a learned codebook. --avg-bits then "
                    "acts as a target rate realized by per-tensor scale calibration."),
    importance_path: Path | None = typer.Option(
        None, help="Activation importance .pt from profile-activations."),
    hessian_path: Path | None = typer.Option(
        None, help="Per-layer Hessian .pt from profile-hessian; enables compensation."),
    rounds: int = typer.Option(3, help="Fit/compensate rounds."),
    compensate: bool = typer.Option(True, help="--no-compensate runs the refit-only control."),
    dataset: str = typer.Option("Salesforce/wikitext"),
    config: str = typer.Option("wikitext-2-raw-v1"),
    max_length: int = typer.Option(4096),
    stride: int = typer.Option(2048),
    out: Path = typer.Option(Path("experiments/bits-per-brain/results.jsonl")),
) -> None:
    """Fake-quantize the expert FFNs (uniform or expert-importance allocation), then measure
    wikitext perplexity; append a row to results.jsonl."""
    import json

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from smart_quant.encode import quantize_experts
    from smart_quant.eval import load_causal_lm, sliding_window_perplexity

    # dataset first, so a bad id fails fast rather than after the multi-minute model load
    tok = AutoTokenizer.from_pretrained(model)
    text = "\n\n".join(load_dataset(dataset, config, split="test")["text"])
    lm = load_causal_lm(model, dtype="auto", device_map="cuda").eval()
    freqs = torch.load(freqs_path, weights_only=True) if allocation == "expert" else None
    importance = torch.load(importance_path, weights_only=True) if importance_path else None
    hessians = torch.load(hessian_path, weights_only=True) if hessian_path else None
    stats = quantize_experts(lm, avg_bits=avg_bits, sub_dim=sub_dim, freqs=freqs, lattice=lattice,
                             bits_lo=bits_lo, bits_hi=bits_hi, codebook_order=codebook_order,
                             importance=importance, hessians=hessians, rounds=rounds,
                             compensate=compensate)
    span = [round(min(s["bits_min"] for s in stats), 2), round(max(s["bits_max"] for s in stats), 2)]

    # Realized footprint: expert_bpw is the honest per-weight cost of the quantized experts
    # (indices + shared codebook) — the quantity to match against imatrix/GGUF targets. model_bpw
    # folds in the still-fp16 non-experts, so it's higher and only comparable to whole-model quants.
    expert_bits = sum(s["quant_bits"] for s in stats)
    expert_weights = sum(s["quant_weights"] for s in stats)
    total_params = sum(p.numel() for p in lm.parameters())
    expert_bpw = expert_bits / expert_weights
    model_bpw = (expert_bits + (total_params - expert_weights) * 16) / total_params
    ppl = sliding_window_perplexity(lm, tok, text, max_length, stride, "cuda")

    row = {"label": label, "model": model, "allocation": allocation, "avg_bits": avg_bits,
           "sub_dim": 8 if lattice else sub_dim, "codebook_order": codebook_order,
           "quantizer": "e8" if lattice else "pq",
           # derived from the artifact's rank, not a hand-typed flag: a mislabeled row would put
           # a data point on the wrong arm of the phase-7 ablation
           "importance": None if importance is None else (
               "expert" if next(iter(importance.values())).dim() == 2 else "layer"),
           "compensation": None if hessians is None else (
               f"rounds={rounds}" if compensate else f"refit-only rounds={rounds}"),
           "wikitext_ppl": round(ppl, 4), "moe_layers": len(stats),
           "per_expert_bits_span": span, "expert_bpw": round(expert_bpw, 3),
           "model_bpw": round(model_bpw, 3)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    console.print(f"[bold]{label}[/bold] ({allocation}, ~{avg_bits}bpw -> {expert_bpw:.3f} expert bpw)  "
                  f"wikitext ppl = [bold]{ppl:.4f}[/bold]  ->  {out}")


def load_for_calibration(model: str) -> tuple[Any, Any, Any, Any]:
    """Tokenizer, cuda model, its text config, and the streaming C4 corpus. Shared by both
    profile commands so they cannot drift onto different calibration data."""
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModel.from_pretrained(model, torch_dtype="auto", device_map="cuda").eval()
    return tok, lm, lm.config.get_text_config(), load_dataset(  # unwraps multimodal text_config
        "allenai/c4", "en", split="train", streaming=True)


def stream_calibration(tok: Any, lm: Any, rows: Any, calib_rows: int, seq_len: int) -> None:
    """Forward `calib_rows` truncated C4 rows through `lm` so registered hooks accumulate."""
    import torch

    for _, row in zip(range(calib_rows), rows):
        ids = tok(row["text"], return_tensors="pt", truncation=True,
                  max_length=seq_len).input_ids.to("cuda")
        with torch.no_grad():
            lm(ids)


@app.command("profile-experts")
def profile_experts(
    model: str = typer.Option(..., help="HF repo id or local path."),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    out: Path = typer.Option(Path("experiments/expert_freq.pt")),
) -> None:
    """Accumulate per-expert selection frequency over a calibration slice."""
    import torch

    from smart_quant.expert_importance import ExpertUsageProfiler

    tok, lm, text_cfg, rows = load_for_calibration(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ExpertUsageProfiler(
        lm, top_k=text_cfg.num_experts_per_tok, num_experts=text_cfg.num_experts
    ) as prof:
        stream_calibration(tok, lm, rows, calib_rows, seq_len)
        freqs = prof.frequencies()
    torch.save(freqs, out)
    console.print(f"profiled {len(freqs)} MoE layers over {calib_rows} rows → {out}")


@app.command("profile-activations")
def profile_activations(
    model: str = typer.Option(..., help="HF repo id or local path."),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    tau: float = typer.Option(1000.0, help="Shrinkage pseudo-count for the per-expert arm."),
    alpha: float = typer.Option(1.0, help="Dynamic-range compression, w**alpha."),
    out_expert: Path = typer.Option(Path("experiments/expert_act_importance_expert.pt")),
    out_layer: Path = typer.Option(Path("experiments/expert_act_importance_layer.pt")),
) -> None:
    """Accumulate per-input-channel E[x^2] for the fused expert projections.

    Writes both granularity arms from one pass — the layer statistic is the token-weighted
    marginal of the per-expert one, so a second calibration run would only re-derive it."""
    import torch

    from smart_quant.expert_importance import (
        ActivationImportanceProfiler, normalize_importance, shrink_importance)

    tok, lm, text_cfg, rows = load_for_calibration(model)
    with ActivationImportanceProfiler(lm, num_experts=text_cfg.num_experts) as prof:
        stream_calibration(tok, lm, rows, calib_rows, seq_len)
        per_expert, per_layer, counts = prof.importance("expert"), prof.importance("layer"), prof.counts

    layer = {k: normalize_importance(v, alpha) for k, v in per_layer.items()}
    expert = {k: normalize_importance(
        shrink_importance(v, counts[k].cpu(), per_layer[k], tau=tau), alpha)
        for k, v in per_expert.items()}
    for path, stats in ((out_expert, expert), (out_layer, layer)):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, path)
    console.print(f"profiled {len(expert)} expert tensors over {calib_rows} rows → "
                  f"{out_expert} (2-D) and {out_layer} (1-D)")


@app.command("profile-hessian")
def profile_hessian(
    model: str = typer.Option(..., help="HF repo id or local path."),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    out: Path = typer.Option(Path("experiments/expert_hessian.pt")),
) -> None:
    """Accumulate the per-layer input second moment for error compensation."""
    import torch

    from smart_quant.expert_importance import HessianProfiler

    tok, lm, _, rows = load_for_calibration(model)
    with HessianProfiler(lm) as prof:
        stream_calibration(tok, lm, rows, calib_rows, seq_len)
        hess = prof.hessians()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hess, out)
    console.print(f"profiled {len(hess)} layer Hessians over {calib_rows} rows → {out}")


if __name__ == "__main__":
    app()
