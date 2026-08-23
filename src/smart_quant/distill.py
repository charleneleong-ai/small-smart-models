"""Knowledge distillation: cache teacher logits, train student against cached targets.

Phase 1: Cache teacher logits (offline KD)
Phase 2: Train student against cached logits
Phase 3: Evaluate distilled student
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CachedLogitsDataset(Dataset):
    """Dataset of cached teacher logits + input IDs for offline distillation."""

    def __init__(
        self,
        cache_dir: Path,
        max_length: int = 2048,
        top_k: int = 100,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self.top_k = top_k

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

        # Sparsify to top-k logits
        if self.top_k < logits.size(-1):
            topk_vals, _ = torch.topk(logits, self.top_k, dim=-1)
            threshold = topk_vals[:, -1:].expand_as(logits)
            mask = logits < threshold
            logits = logits.masked_fill(mask, float("-inf"))

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
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    # Load dataset
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Cache logits
    index_entries = []
    shard_idx = 0
    shard_data = {"input_ids": [], "logits": []}

    for i, sample in enumerate(ds):
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
            logits = outputs.logits.float()  # Keep in fp32 for precision

        # Sparsify to top-k
        if top_k < logits.size(-1):
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold = topk_vals[:, :, -1:].expand_as(logits)
            mask = logits < threshold
            logits = logits.masked_fill(mask, float("-inf"))

        shard_data["input_ids"].append(enc["input_ids"].cpu())
        shard_data["logits"].append(logits.cpu())

        # Save shard when full
        if len(shard_data["input_ids"]) >= shard_size:
            shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
            torch.save({
                "input_ids": torch.cat(shard_data["input_ids"], dim=0),
                "logits": torch.cat(shard_data["logits"], dim=0),
            }, shard_path)

            for j in range(len(shard_data["input_ids"])):
                index_entries.append({
                    "shard": shard_idx,
                    "offset": j,
                    "length": shard_data["input_ids"][j].size(0),
                })

            shard_data = {"input_ids": [], "logits": []}
            shard_idx += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(ds)} samples, {shard_idx} shards saved")

    # Save final partial shard
    if shard_data["input_ids"]:
        shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
        torch.save({
            "input_ids": torch.cat(shard_data["input_ids"], dim=0),
            "logits": torch.cat(shard_data["logits"], dim=0),
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
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load student
    tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        student_id,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load dataset
    dataset = CachedLogitsDataset(cache_dir, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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
                   config={"student": student_id, "temperature": temperature, "alpha": alpha,
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
