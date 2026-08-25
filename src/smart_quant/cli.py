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
    tasks: str = typer.Option(None, help="Comma-separated lm-eval tasks for the capability "
                                         "battery (e.g. arc_challenge,gsm8k); appends task_acc."),
    limit: int = typer.Option(None, help="Per-task sample limit for the battery (None = full)."),
    out: Path = typer.Option(Path("experiments/bits-per-brain/results.jsonl")),
    wandb: bool = typer.Option(False, help="Log metrics to Weights & Biases."),
    wandb_project: str = typer.Option("small-smart-models", help="W&B project name."),
) -> None:
    """Sliding-window wikitext perplexity for one model; append a row to results.jsonl.

    With --tasks, also runs the lm-eval capability battery on the same in-memory model and
    appends `task_acc` to the row — the footprint-matched artifact its ppl came from."""
    import json

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from smart_quant.eval import load_causal_lm, run_task_battery, sliding_window_perplexity

    # dataset first, so a bad id fails fast rather than after the multi-minute model load
    # A GGUF repo's own tokenizer is not necessarily byte-identical to the base model's, and
    # perplexity is only comparable across rows when the token count is. Default to --model so
    # existing calls are unchanged; override to the fp16 base when scoring a GGUF.
    tok = AutoTokenizer.from_pretrained(tokenizer or model)
    text = "\n\n".join(load_dataset(dataset, config, split="test")["text"])
    load_kwargs = {"dtype": "auto", "device_map": "auto"}
    if gguf_file:
        # GGUF is dequantized on load, so a 35B build lands at ~70 GB in fp16 — close enough to an
        # 80 GB card that "auto" should be free to spill rather than OOM. fp32 would not fit at all.
        load_kwargs |= {"gguf_file": gguf_file, "dtype": "float16", "device_map": "auto"}
    lm = load_causal_lm(model, **load_kwargs).eval()
    ppl = sliding_window_perplexity(lm, tok, text, max_length, stride, "cuda")

    task_acc = None
    if tasks:
        task_acc = run_task_battery(lm, tok, [t.strip() for t in tasks.split(",")], limit)

    row = {"label": label, "model": model, "gguf_file": gguf_file,
           "tokenizer": tokenizer or model, "wikitext_ppl": round(ppl, 4),
           "dataset": f"{dataset}:{config}"}
    if task_acc:
        row["task_acc"] = task_acc
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    console.print(f"[bold]{label}[/bold]  wikitext-2 ppl = [bold]{ppl:.4f}[/bold]  ->  {out}")

    if wandb:
        import trackio
        trackio.init(project=wandb_project, name=label, config=row)
        log_row = {"wikitext_ppl": round(ppl, 4)}
        if task_acc:
            log_row.update({f"acc/{k}": v for k, v in task_acc.items()})
        trackio.log(log_row)
        trackio.finish()


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
    noise: bool = typer.Option(
        False, help="Inject Gaussian noise matched to PQ's perturbation RMS (ablation baseline)."),
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
    tasks: str = typer.Option(None, help="Comma-separated lm-eval tasks for the capability "
                                         "battery (e.g. arc_challenge,gsm8k); appends task_acc."),
    limit: int = typer.Option(None, help="Per-task sample limit for the battery (None = full)."),
    out: Path = typer.Option(Path("experiments/bits-per-brain/results.jsonl")),
    wandb: bool = typer.Option(False, help="Log metrics to Weights & Biases."),
    wandb_project: str = typer.Option("small-smart-models", help="W&B project name."),
) -> None:
    """Fake-quantize the expert FFNs (uniform or expert-importance allocation), then measure
    wikitext perplexity; append a row to results.jsonl.

    With --tasks, also runs the lm-eval capability battery on the quantized model in memory —
    task accuracy is the "still smart" half of the footprint-matched claim."""
    import json

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from smart_quant.encode import quantize_experts, noise_inject_experts
    from smart_quant.eval import load_causal_lm, run_task_battery, sliding_window_perplexity

    # dataset first, so a bad id fails fast rather than after the multi-minute model load
    tok = AutoTokenizer.from_pretrained(model)
    text = "\n\n".join(load_dataset(dataset, config, split="test")["text"])
    lm = load_causal_lm(model, dtype="auto", device_map="auto").eval()
    freqs = torch.load(freqs_path, weights_only=True) if allocation == "expert" else None
    importance = torch.load(importance_path, weights_only=True) if importance_path else None
    hessians = torch.load(hessian_path, weights_only=True) if hessian_path else None
    if noise:
        stats = noise_inject_experts(lm, avg_bits=avg_bits, sub_dim=sub_dim)
    else:
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

    task_acc = None
    if tasks:
        task_acc = run_task_battery(lm, tok, [t.strip() for t in tasks.split(",")], limit)

    row = {"label": label, "model": model, "allocation": allocation, "avg_bits": avg_bits,
           "sub_dim": 8 if lattice else sub_dim, "codebook_order": codebook_order,
           "quantizer": "e8" if lattice else ("noise" if noise else "pq"),
           # derived from the artifact's rank, not a hand-typed flag: a mislabeled row would put
           # a data point on the wrong arm of the phase-7 ablation
           "importance": None if importance is None else (
               "expert" if next(iter(importance.values())).dim() == 2 else "layer"),
           "compensation": None if hessians is None else (
               f"rounds={rounds}" if compensate else f"refit-only rounds={rounds}"),
           "wikitext_ppl": round(ppl, 4), "moe_layers": len(stats),
           "per_expert_bits_span": span, "expert_bpw": round(expert_bpw, 3),
           "model_bpw": round(model_bpw, 3)}
    if task_acc:
        row["task_acc"] = task_acc
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    console.print(f"[bold]{label}[/bold] ({allocation}, ~{avg_bits}bpw -> {expert_bpw:.3f} expert bpw)  "
                  f"wikitext ppl = [bold]{ppl:.4f}[/bold]  ->  {out}")

    if wandb:
        import trackio
        trackio.init(project=wandb_project, name=label, config=row)
        log_row = {
            "wikitext_ppl": round(ppl, 4),
            "expert_bpw": round(expert_bpw, 3),
            "model_bpw": round(model_bpw, 3),
        }
        if task_acc:
            log_row.update({f"acc/{k}": v for k, v in task_acc.items()})
        trackio.log(log_row)
        trackio.finish()


def load_for_calibration(model: str) -> tuple[Any, Any, Any, Any]:
    """Tokenizer, cuda model, its text config, and the streaming C4 corpus. Shared by both
    profile commands so they cannot drift onto different calibration data."""
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModel.from_pretrained(model, torch_dtype="auto", device_map="auto").eval()
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


@app.command("cache-teacher")
def cache_teacher_cmd(
    model: str = typer.Option("Qwen/Qwen3.8-27B", help="Teacher model HF repo id or local path."),
    out: Path = typer.Option(Path("experiments/teacher_logits"), help="Output directory for cached logits."),
    dataset: str = typer.Option("allenai/c4", help="HF dataset repo id."),
    config: str = typer.Option("en", help="Dataset config."),
    split: str = typer.Option("train", help="Dataset split."),
    max_length: int = typer.Option(2048),
    shard_size: int = typer.Option(1000),
    top_k: int = typer.Option(100, help="Keep only top-K logits per token."),
    max_samples: int = typer.Option(512, help="Max samples to process."),
) -> None:
    """Cache teacher model's top-K logits for offline distillation.

    Phase 1 of the distillation pipeline. Runs the teacher over a calibration set
    and caches the sparse logits to disk for student training.
    """
    from smart_quant.distill import cache_teacher_logits

    console.print(f"[bold]Caching teacher logits from {model}[/bold]")
    console.print(f"  Dataset: {dataset}:{config} ({split})")
    console.print(f"  Max samples: {max_samples}")
    console.print(f"  Top-K: {top_k}")

    cache_teacher_logits(
        model_id=model,
        output_dir=out,
        dataset_name=dataset,
        dataset_config=config,
        split=split,
        max_length=max_length,
        shard_size=shard_size,
        top_k=top_k,
        max_samples=max_samples,
    )

    console.print(f"[bold green]Teacher logits cached to {out}[/bold green]")


@app.command("distill")
def distill_cmd(
    student: str = typer.Option("Qwen/Qwen3-0.6B", help="Student model HF repo id or local path."),
    teacher_model: str = typer.Option("Qwen/Qwen3.8-27B", help="Teacher model ID for vocab projection."),
    cache_dir: Path = typer.Option(Path("experiments/teacher_logits"), help="Directory with cached teacher logits."),
    out: Path = typer.Option(Path("experiments/distilled-models/student-1.5b"), help="Output directory."),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(4),
    learning_rate: float = typer.Option(5e-5),
    temperature: float = typer.Option(4.0, help="Distillation temperature."),
    alpha: float = typer.Option(0.7, help="Weight for distillation loss (1-alpha for hard labels)."),
    max_length: int = typer.Option(2048),
    warmup_ratio: float = typer.Option(0.1),
    gradient_accumulation_steps: int = typer.Option(4),
    save_steps: int = typer.Option(500),
    logging_steps: int = typer.Option(50),
    wandb: bool = typer.Option(False, help="Log metrics to Weights & Biases."),
    wandb_project: str = typer.Option("small-smart-models"),
) -> None:
    """Train student model against cached teacher logits.

    Phase 2 of the distillation pipeline. Loads cached teacher logits and trains
    a smaller student model using knowledge distillation.
    """
    from smart_quant.distill import train_student

    console.print(f"[bold]Training student {student} with distillation[/bold]")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Alpha: {alpha}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  LR: {learning_rate}")

    train_student(
        student_id=student,
        cache_dir=cache_dir,
        output_dir=out,
        teacher_model_id=teacher_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        alpha=alpha,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        wandb=wandb,
        wandb_project=wandb_project,
    )

    console.print(f"[bold green]Student trained and saved to {out}[/bold green]")


@app.command("distill-eval")
def distill_eval_cmd(
    model: str = typer.Option(..., help="Path to distilled student model."),
    teacher: str = typer.Option("Qwen/Qwen3.8-27B", help="Teacher model for comparison."),
    label: str = typer.Option(..., help="Row label for results.jsonl."),
    tasks: str = typer.Option("arc_challenge,hellaswag,winogrande,gsm8k,mmlu",
                              help="Comma-separated lm-eval tasks."),
    limit: int = typer.Option(None, help="Per-task sample limit."),
    dataset: str = typer.Option("Salesforce/wikitext"),
    config: str = typer.Option("wikitext-2-raw-v1"),
    max_length: int = typer.Option(4096),
    stride: int = typer.Option(2048),
    out: Path = typer.Option(Path("experiments/distillation/results.jsonl")),
    wandb: bool = typer.Option(False, help="Log metrics to Weights & Biases."),
    wandb_project: str = typer.Option("small-smart-models"),
) -> None:
    """Evaluate a distilled student model against its teacher.

    Phase 3 of the distillation pipeline. Runs the capability battery on the
    student and compares with teacher performance.
    """
    import json

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from smart_quant.eval import load_causal_lm, run_task_battery, sliding_window_perplexity

    console.print(f"[bold]Evaluating distilled model: {model}[/bold]")

    # Load student
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    text = "\n\n".join(load_dataset(dataset, config, split="test")["text"])
    lm = load_causal_lm(model, dtype="auto", device_map="auto").eval()

    # Evaluate
    ppl = sliding_window_perplexity(lm, tok, text, max_length, stride, "cuda")
    task_acc = run_task_battery(lm, tok, [t.strip() for t in tasks.split(",")], limit)

    # Save results
    row = {
        "label": label,
        "model": model,
        "teacher": teacher,
        "wikitext_ppl": round(ppl, 4),
        "dataset": f"{dataset}:{config}",
        "task_acc": task_acc,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")

    console.print(f"[bold]{label}[/bold]  wikitext-2 ppl = [bold]{ppl:.4f}[/bold]")
    for task, acc in task_acc.items():
        console.print(f"  {task}: {acc:.4f}")

    if wandb:
        import trackio
        trackio.init(project=wandb_project, name=label, config=row)
        log_row = {"wikitext_ppl": round(ppl, 4)}
        log_row.update({f"acc/{k}": v for k, v in task_acc.items()})
        trackio.log(log_row)
        trackio.finish()


if __name__ == "__main__":
    app()
