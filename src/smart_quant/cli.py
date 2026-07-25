"""smart-quant CLI — footprint accounting, perplexity smoke test, expert profiling."""
from __future__ import annotations

from pathlib import Path

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
    tok = AutoTokenizer.from_pretrained(model)
    text = "\n\n".join(load_dataset(dataset, config, split="test")["text"])
    load_kwargs = {"dtype": "auto", "device_map": "cuda"}
    if gguf_file:
        load_kwargs["gguf_file"] = gguf_file
    lm = load_causal_lm(model, **load_kwargs).eval()
    ppl = sliding_window_perplexity(lm, tok, text, max_length, stride, "cuda")

    row = {"label": label, "model": model, "gguf_file": gguf_file,
           "wikitext_ppl": round(ppl, 4), "dataset": f"{dataset}:{config}"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    console.print(f"[bold]{label}[/bold]  wikitext-2 ppl = [bold]{ppl:.4f}[/bold]  ->  {out}")


@app.command("profile-experts")
def profile_experts(
    model: str = typer.Option(..., help="HF repo id or local path."),
    calib_rows: int = typer.Option(512),
    seq_len: int = typer.Option(2048),
    out: Path = typer.Option(Path("experiments/expert_freq.pt")),
) -> None:
    """Accumulate per-expert selection frequency over a calibration slice."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    from smart_quant.expert_importance import ExpertUsageProfiler

    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModel.from_pretrained(model, torch_dtype="auto", device_map="cuda").eval()
    text_cfg = lm.config.get_text_config()  # unwraps the multimodal text_config
    rows = load_dataset("allenai/c4", "en", split="train", streaming=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with ExpertUsageProfiler(
        lm, top_k=text_cfg.num_experts_per_tok, num_experts=text_cfg.num_experts
    ) as prof:
        for i, row in zip(range(calib_rows), rows):
            ids = tok(row["text"], return_tensors="pt", truncation=True,
                      max_length=seq_len).input_ids.to("cuda")
            with torch.no_grad():
                lm(ids)
        freqs = prof.frequencies()
    torch.save(freqs, out)
    console.print(f"profiled {len(freqs)} MoE layers over {calib_rows} rows → {out}")


if __name__ == "__main__":
    app()
