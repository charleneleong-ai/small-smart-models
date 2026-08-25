"""Knowledge distillation: cache teacher logits, train student against cached targets.

Phase 1: Cache teacher logits (offline KD)
Phase 2: Train student against cached logits (with vocab projection if teacher != student vocab)
Phase 3: Evaluate distilled student
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def build_vocab_projection(
    teacher_model_id: str,
    student_model_id: str,
    device: str = "cpu",
) -> torch.Tensor:
    """Build a teacher→student vocab mapping as a 1-D lookup tensor.

    For each teacher token, decode it to text and re-encode with the student
    tokenizer. The resulting student token ID becomes the mapping target.
    Teacher tokens that map to multiple student tokens are assigned to the
    first sub-token (approximate but preserves most probability mass).

    Returns:
        Long tensor of shape [teacher_vocab] where entry[i] = student token ID.
    """
    from transformers import AutoTokenizer

    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
    student_tok = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)

    teacher_vocab = teacher_tok.vocab_size
    student_vocab = student_tok.vocab_size

    mapping = torch.zeros(teacher_vocab, dtype=torch.long)
    mapped = 0
    for t_id in range(teacher_vocab):
        token_str = teacher_tok.decode([t_id])
        s_ids = student_tok.encode(token_str, add_special_tokens=False)
        if s_ids:
            mapping[t_id] = s_ids[0]
            mapped += 1

    print(f"Vocab projection: {mapped}/{teacher_vocab} teacher tokens mapped to student vocab")
    return mapping


class CachedLogitsDataset(Dataset):
    """Dataset of cached teacher logits + input IDs for offline distillation."""

    def __init__(
        self,
        cache_dir: Path,
        max_length: int = 2048,
        top_k: int = 100,
        vocab_proj: torch.Tensor | None = None,
        student_vocab_size: int | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self.top_k = top_k
        self.vocab_proj = vocab_proj  # [teacher_vocab] long mapping or None
        self.student_vocab_size = student_vocab_size

        # Load index file
        index_path = self.cache_dir / "index.jsonl"
        self.index: list[dict[str, Any]] = []
        with open(index_path) as f:
            for line in f:
                self.index.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.index[idx]
        shard = entry["shard"]

        # Load cached shard
        shard_path = self.cache_dir / f"shard_{shard:04d}.pt"
        data = torch.load(shard_path, weights_only=True)

        offset = entry["offset"]
        length = min(entry["length"], self.max_length)

        # Index into batch dimension to get single sample
        input_ids = data["input_ids"][offset, :length]
        logits = data["logits"][offset, :length, :]

        # Project teacher logits AND input_ids to student vocab if needed
        if self.vocab_proj is not None:
            # vocab_proj: [teacher_vocab] long tensor mapping teacher→student token IDs
            proj = self.vocab_proj

            # Remap input_ids: teacher token ID → student token ID
            safe_ids = input_ids.clamp(0, proj.size(0) - 1)
            input_ids = proj[safe_ids]

            # Remap logits: scatter teacher logits into student vocab
            teacher_logits_vocab = logits.size(-1)
            student_vocab = self.student_vocab_size or (int(proj.max()) + 1)
            projected = torch.full(logits.shape[:-1] + (student_vocab,), float("-inf"), dtype=logits.dtype)
            proj_expanded = proj.to(logits.device)
            if proj_expanded.size(0) < teacher_logits_vocab:
                pad = torch.zeros(teacher_logits_vocab - proj_expanded.size(0), dtype=proj_expanded.dtype, device=proj_expanded.device)
                proj_expanded = torch.cat([proj_expanded, pad])
            projected.scatter_add_(-1, proj_expanded.expand_as(logits), logits)
            # Replace -inf with large negative for numerically stable softmax
            projected = projected.masked_fill(projected == float("-inf"), -1e4)
            logits = projected
        else:
            # Sparsify to top-k logits (only when no projection)
            if self.top_k < logits.size(-1):
                topk_vals, _ = torch.topk(logits, self.top_k, dim=-1)
                threshold = topk_vals[:, -1:].expand_as(logits)
                mask = logits < threshold
                logits = logits.masked_fill(mask, float("-inf"))

        # Pad to max_length so DataLoader can collate into batches
        pad_len = self.max_length - input_ids.size(0)
        if pad_len > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
            logits = torch.nn.functional.pad(logits, (0, 0, 0, pad_len), value=float("-inf"))

        return {
            "input_ids": input_ids,
            "logits": logits,
        }


def cache_teacher_logits(
    model_id: str,
    output_dir: Path,
    dataset_name: str = "c4",
    dataset_config: str = "en",
    split: str = "train",
    max_length: int = 2048,
    shard_size: int = 1000,
    top_k: int = 100,
    max_samples: int | None = None,
    device: str = "cuda",
) -> Path:
    """Cache teacher model's top-K logits for offline distillation.

    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Where to save cached logits
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split
        max_length: Max sequence length per sample
        shard_size: Number of samples per shard file
        top_k: Keep only top-K logits per token
        max_samples: Max samples to process (None = all)
        device: Device to run on

    Returns:
        Path to output directory
    """
    from itertools import islice

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from smart_quant.eval import load_causal_lm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher using the fallback-aware loader
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_causal_lm(model_id, dtype=torch.float16, device_map=device,
                           trust_remote_code=True, low_cpu_mem_usage=True).eval()

    # Load dataset (streaming to avoid full download)
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    # Cache logits
    index_entries = []
    shard_idx = 0
    shard_data: dict[str, list[torch.Tensor]] = {"input_ids": [], "logits": []}

    # Use islice for streaming mode to limit samples
    sample_iter = islice(ds, max_samples) if max_samples else ds

    for i, sample in enumerate(sample_iter):
        text = sample.get("text", sample.get("content", ""))
        if not text:
            continue

        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits.half()  # fp16 to save CPU RAM

        # Sparsify to top-k
        if top_k < logits.size(-1):
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold = topk_vals[:, :, -1:].expand_as(logits)
            mask = logits < threshold
            logits = logits.masked_fill(mask, float("-inf"))

        shard_data["input_ids"].append(enc["input_ids"].cpu().squeeze(0))  # [seq_len]
        shard_data["logits"].append(logits.cpu().squeeze(0))              # [seq_len, vocab]
        del enc, outputs, logits
        torch.cuda.empty_cache()

        # Save shard when full or every 100 samples to cap RAM
        effective_shard = min(shard_size, 100) if max_samples and max_samples < shard_size else shard_size
        if len(shard_data["input_ids"]) >= effective_shard:
            shard_path = output_dir / f"shard_{shard_idx:04d}.pt"

            # Pad to max_length in this batch for uniform tensor sizes
            batch_max = max(x.size(0) for x in shard_data["input_ids"])
            padded_ids = torch.zeros(len(shard_data["input_ids"]), batch_max, dtype=torch.long)
            padded_logits = torch.zeros(len(shard_data["logits"]), batch_max, shard_data["logits"][0].size(-1), dtype=torch.float16)
            for j, (ids, lg) in enumerate(zip(shard_data["input_ids"], shard_data["logits"])):
                padded_ids[j, :ids.size(0)] = ids
                padded_logits[j, :lg.size(0)] = lg

            torch.save({
                "input_ids": padded_ids,
                "logits": padded_logits,
            }, shard_path)

            for j in range(len(shard_data["input_ids"])):
                index_entries.append({
                    "shard": shard_idx,
                    "offset": j,
                    "length": shard_data["input_ids"][j].size(0),
                })

            shard_data = {"input_ids": [], "logits": []}
            shard_idx += 1
            gc.collect()

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples, {shard_idx} shards saved")

    # Save final partial shard
    if shard_data["input_ids"]:
        shard_path = output_dir / f"shard_{shard_idx:04d}.pt"

        batch_max = max(x.size(0) for x in shard_data["input_ids"])
        padded_ids = torch.zeros(len(shard_data["input_ids"]), batch_max, dtype=torch.long)
        padded_logits = torch.zeros(len(shard_data["logits"]), batch_max, shard_data["logits"][0].size(-1), dtype=torch.float16)
        for j, (ids, lg) in enumerate(zip(shard_data["input_ids"], shard_data["logits"])):
            padded_ids[j, :ids.size(0)] = ids
            padded_logits[j, :lg.size(0)] = lg

        torch.save({
            "input_ids": padded_ids,
            "logits": padded_logits,
        }, shard_path)

        for j in range(len(shard_data["input_ids"])):
            index_entries.append({
                "shard": shard_idx,
                "offset": j,
                "length": shard_data["input_ids"][j].size(0),
            })

    # Save index
    with open(output_dir / "index.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in index_entries)

    # Save metadata
    meta = {
        "model_id": model_id,
        "dataset": f"{dataset_name}:{dataset_config}",
        "split": split,
        "max_length": max_length,
        "top_k": top_k,
        "total_samples": len(index_entries),
        "total_shards": shard_idx + 1,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Cached {len(index_entries)} samples to {output_dir}")
    return output_dir


def train_student(
    student_id: str,
    cache_dir: Path,
    output_dir: Path,
    teacher_model_id: str | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    temperature: float = 4.0,
    alpha: float = 0.7,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 1000,
    logging_steps: int = 100,
    device: str = "cuda",
    wandb: bool = False,
    wandb_project: str = "small-smart-models",
) -> Path:
    """Train student model against cached teacher logits.

    Args:
        student_id: HuggingFace model ID or local path for student
        cache_dir: Directory with cached teacher logits
        output_dir: Where to save trained student
        teacher_model_id: Teacher model ID (for vocab projection if vocab differs)
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Peak learning rate
        temperature: Distillation temperature
        alpha: Weight for distillation loss (1-alpha for hard labels)
        max_length: Max sequence length
        warmup_ratio: Fraction of steps for LR warmup
        weight_decay: Weight decay coefficient
        gradient_accumulation_steps: Accumulate gradients over N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        device: Device to train on
        wandb: Enable W&B logging
        wandb_project: W&B project name

    Returns:
        Path to output directory
    """
    from transformers import (
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from smart_quant.eval import load_causal_lm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load student using the fallback-aware loader
    tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    model = load_causal_lm(student_id, dtype=torch.float16, device_map=device,
                           trust_remote_code=True)

    # Build vocab projection if teacher and student have different vocabs
    vocab_proj = None
    saved_proj_path = cache_dir / "vocab_proj.pt"
    if saved_proj_path.exists():
        vocab_proj = torch.load(saved_proj_path, weights_only=True)
        print(f"Loaded saved vocab projection from {saved_proj_path}")
    elif teacher_model_id:
        teacher_tok = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
        if teacher_tok.vocab_size != tokenizer.vocab_size:
            print(f"Building vocab projection: teacher {teacher_tok.vocab_size} → student {tokenizer.vocab_size}")
            vocab_proj = build_vocab_projection(teacher_model_id, student_id, device="cpu")
            torch.save(vocab_proj, saved_proj_path)
            print(f"Saved vocab projection to {saved_proj_path}")

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load dataset
    student_vocab_size = model.config.vocab_size
    dataset = CachedLogitsDataset(cache_dir, max_length=max_length, vocab_proj=vocab_proj,
                                  student_vocab_size=student_vocab_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Scheduler
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # W&B
    if wandb:
        import wandb
        wandb.init(project=wandb_project, name=f"distill-{student_id.split('/')[-1]}",
                   config={"student": student_id, "teacher": teacher_model_id,
                           "temperature": temperature, "alpha": alpha,
                           "lr": learning_rate, "epochs": epochs})

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            teacher_logits = batch["logits"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids)
            student_logits = outputs.logits

            # Distillation loss (KL divergence with temperature)
            T = temperature
            loss_kl = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction="batchmean",
                log_target=False,
            ) * (T ** 2)

            # Hard label loss (cross-entropy on input_ids shifted by 1)
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            # Combined loss
            loss = alpha * loss_kl + (1 - alpha) * loss_ce

            # Backward
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                running_loss += loss.item() * gradient_accumulation_steps

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

                    if wandb:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch + batch_idx / len(dataloader),
                        })

                    running_loss = 0.0

                if global_step % save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"  Saved checkpoint to {ckpt_dir}")

        print(f"Epoch {epoch + 1}/{epochs} completed")

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config
    config = {
        "student_id": student_id,
        "teacher_model_id": teacher_model_id,
        "cache_dir": str(cache_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "alpha": alpha,
        "max_length": max_length,
        "total_steps": global_step,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if wandb:
        import wandb
        wandb.finish()

    print(f"Training complete. Model saved to {output_dir}")
    return output_dir
