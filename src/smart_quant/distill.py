"""Knowledge distillation: cache teacher logits, train student against cached targets.

Phase 1: Cache teacher logits (offline KD) — sparse top-k format in TEACHER vocab
Phase 2: Train student against cached logits — project STUDENT logits UP to teacher vocab
Phase 3: Evaluate distilled student

Sparse format: each shard stores top-k (values, indices) per position instead of
the full vocab distribution. Reduces shard size from ~15GB to ~150MB.

Cross-vocab strategy: teacher logits stay in teacher vocab. During training,
student logits are projected UP to teacher vocab via a learned projection matrix.
KL divergence computed entirely in teacher vocab space — no -inf positions.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class VocabProjector(torch.nn.Module):
    """Projects student logits to teacher vocab space via scatter lookup.

    Instead of a dense [student_vocab × teacher_vocab] matrix (143GB for this pair),
    stores a sparse index mapping [student_vocab] → teacher_token_id and uses scatter_
    to route logits.
    """

    def __init__(self, student_to_teacher: torch.Tensor, teacher_vocab: int) -> None:
        super().__init__()
        self.register_buffer("mapping", student_to_teacher.long())  # [student_vocab]
        self.teacher_vocab = teacher_vocab

    def forward(self, student_logits: torch.Tensor) -> torch.Tensor:
        """Project [B, S, student_vocab] → [B, S, teacher_vocab]."""
        B, S, _ = student_logits.shape
        # Initialize with -inf (unmapped positions)
        teacher_logits = torch.full((B, S, self.teacher_vocab), float("-inf"),
                                    dtype=student_logits.dtype, device=student_logits.device)
        # Expand mapping to batch: [B, S, student_vocab]
        expanded_map = self.mapping.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        teacher_logits.scatter_(-1, expanded_map, student_logits)
        return teacher_logits


def build_reverse_vocab_projection(
    teacher_model_id: str,
    student_model_id: str,
    student_model_vocab_size: int | None = None,
) -> torch.Tensor:
    """Build a student→teacher vocab mapping.

    Args:
        teacher_model_id: Teacher model ID
        student_model_id: Student model ID
        student_model_vocab_size: Model's actual vocab size (may exceed tokenizer.vocab_size)

    Returns:
        Long tensor of shape [student_model_vocab_size] where entry[i] = teacher token ID.
    """
    from transformers import AutoTokenizer

    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
    student_tok = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)

    # Use model's vocab size (may be larger than tokenizer.vocab_size)
    student_vocab = student_model_vocab_size or student_tok.vocab_size
    mapping = torch.zeros(student_vocab, dtype=torch.long)
    mapped = 0
    for s_id in range(min(student_tok.vocab_size, student_vocab)):
        token_str = student_tok.decode([s_id])
        t_ids = teacher_tok.encode(token_str, add_special_tokens=False)
        if t_ids:
            mapping[s_id] = t_ids[0]
            mapped += 1

    print(f"Reverse vocab projection: {mapped}/{student_vocab} student tokens mapped to teacher vocab")
    return mapping


def build_vocab_projection(
    teacher_model_id: str,
    student_model_id: str,
    device: str = "cpu",
) -> torch.Tensor:
    """Build a teacher→student vocab mapping as a 1-D lookup tensor.

    Returns:
        Long tensor of shape [teacher_vocab] where entry[i] = student token ID.
    """
    from transformers import AutoTokenizer

    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
    student_tok = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)

    teacher_vocab = teacher_tok.vocab_size
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
    """Dataset of cached teacher logits (sparse top-k format) in teacher vocab space.

    Each shard stores:
        input_ids:  [N, S] long (teacher tokenizer — unused at training time)
        logit_values:  [N, S, k] fp16  (top-k values)
        logit_indices: [N, S, k] int32 (top-k positions in TEACHER vocab)
        text: list[str] — original text, re-tokenized with student tokenizer at load time
    """

    def __init__(
        self,
        cache_dir: Path,
        max_length: int = 2048,
        student_tokenizer=None,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self.student_tokenizer = student_tokenizer

        index_path = self.cache_dir / "index.jsonl"
        self.index: list[dict[str, Any]] = []
        with open(index_path) as f:
            for line in f:
                self.index.append(json.loads(line))

        self._shard_cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_id: int) -> dict[str, torch.Tensor]:
        if shard_id not in self._shard_cache:
            shard_path = self.cache_dir / f"shard_{shard_id:04d}.pt"
            self._shard_cache[shard_id] = torch.load(shard_path, weights_only=False)
        return self._shard_cache[shard_id]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.index[idx]
        shard_id = entry["shard"]
        data = self._load_shard(shard_id)

        offset = entry["offset"]
        length = min(entry["length"], self.max_length)

        # Re-tokenize with student tokenizer so input_ids are in student vocab
        if self.student_tokenizer is not None and "text" in data:
            text = data["text"][offset]
            enc = self.student_tokenizer(
                text, max_length=self.max_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)   # [max_length]
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            # Fallback: use cached input_ids (may be teacher vocab)
            input_ids = data["input_ids"][offset, :self.max_length]
            attention_mask = (input_ids != 0).long()
            pad_len = self.max_length - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        logit_values = data["logit_values"][offset, :length, :]    # [length, k]
        logit_indices = data["logit_indices"][offset, :length, :]  # [length, k]

        # Pad logits to max_length
        pad_len = self.max_length - logit_values.size(0)
        if pad_len > 0:
            logit_values = torch.nn.functional.pad(logit_values, (0, 0, 0, pad_len), value=float("-inf"))
            logit_indices = torch.nn.functional.pad(logit_indices, (0, 0, 0, pad_len), value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "logit_values": logit_values,
            "logit_indices": logit_indices,
        }


def cache_teacher_logits(
    model_id: str,
    output_dir: Path,
    dataset_name: str = "c4",
    dataset_config: str = "en",
    split: str = "train",
    max_length: int = 2048,
    shard_size: int = 100,
    top_k: int = 100,
    max_samples: int | None = None,
    device: str = "cuda",
) -> Path:
    """Cache teacher model's top-K logits in sparse format (teacher vocab space).

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

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_causal_lm(model_id, dtype=torch.float16, device_map=device,
                           trust_remote_code=True, low_cpu_mem_usage=True).eval()

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    index_entries = []
    shard_idx = 0
    shard_data: dict[str, list[torch.Tensor | str]] = {
        "input_ids": [], "logit_values": [], "logit_indices": [], "text": [],
    }

    sample_iter = islice(ds, max_samples) if max_samples else ds

    for i, sample in enumerate(sample_iter):
        text = sample.get("text", sample.get("content", ""))
        if not text:
            continue

        enc = tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            full_logits = outputs.logits.float()  # [1, seq_len, vocab]

        # Extract top-k sparse representation (in teacher vocab space)
        k = min(top_k, full_logits.size(-1))
        topk_vals, topk_idx = torch.topk(full_logits, k, dim=-1)  # [1, S, k]

        shard_data["input_ids"].append(enc["input_ids"].cpu().squeeze(0))
        shard_data["logit_values"].append(topk_vals.cpu().squeeze(0).half())   # [S, k] fp16
        shard_data["logit_indices"].append(topk_idx.cpu().squeeze(0).int())    # [S, k] int32
        shard_data["text"].append(text)

        del enc, outputs, full_logits, topk_vals, topk_idx
        torch.cuda.empty_cache()

        if len(shard_data["input_ids"]) >= shard_size:
            _save_sparse_shard(output_dir, shard_idx, shard_data, index_entries)
            shard_data = {"input_ids": [], "logit_values": [], "logit_indices": [], "text": []}
            shard_idx += 1
            gc.collect()

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples, {shard_idx} shards saved")

    if shard_data["input_ids"]:
        _save_sparse_shard(output_dir, shard_idx, shard_data, index_entries)

    with open(output_dir / "index.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in index_entries)

    meta = {
        "model_id": model_id,
        "teacher_vocab_size": model.config.vocab_size,
        "dataset": f"{dataset_name}:{dataset_config}",
        "split": split,
        "max_length": max_length,
        "top_k": top_k,
        "format": "sparse_topk",
        "total_samples": len(index_entries),
        "total_shards": shard_idx + 1,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Cached {len(index_entries)} samples (sparse) to {output_dir}")
    return output_dir


def _save_sparse_shard(
    output_dir: Path,
    shard_idx: int,
    shard_data: dict[str, list[torch.Tensor | str]],
    index_entries: list[dict[str, Any]],
) -> None:
    """Pad and save a sparse shard."""
    n = len(shard_data["input_ids"])
    seq_max = max(x.size(0) for x in shard_data["input_ids"])
    k = shard_data["logit_values"][0].size(-1)

    padded_ids = torch.zeros(n, seq_max, dtype=torch.long)
    padded_vals = torch.full((n, seq_max, k), float("-inf"), dtype=torch.float16)
    padded_idx = torch.zeros(n, seq_max, k, dtype=torch.int32)

    for j in range(n):
        s = shard_data["input_ids"][j].size(0)
        padded_ids[j, :s] = shard_data["input_ids"][j]
        padded_vals[j, :s] = shard_data["logit_values"][j]
        padded_idx[j, :s] = shard_data["logit_indices"][j]

    shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
    torch.save({
        "input_ids": padded_ids,
        "logit_values": padded_vals,
        "logit_indices": padded_idx,
        "text": shard_data["text"],
    }, shard_path)

    for j in range(n):
        index_entries.append({
            "shard": shard_idx,
            "offset": j,
            "length": int(shard_data["input_ids"][j].size(0)),
        })


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

    Cross-vocab: student logits are projected UP to teacher vocab space via
    a learned VocabProjector. KL divergence computed in teacher vocab space.

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

    tokenizer = AutoTokenizer.from_pretrained(student_id, trust_remote_code=True)
    model = load_causal_lm(student_id, dtype=torch.float16, device_map=device,
                           trust_remote_code=True)
    student_vocab_size = model.config.vocab_size

    # Build vocab projector: student → teacher vocab
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    teacher_tok = AutoTokenizer.from_pretrained(teacher_model_id or meta.get("model_id", student_id),
                                                trust_remote_code=True)
    teacher_vocab_size = teacher_tok.vocab_size

    # Check meta.json for teacher model vocab size
    if meta.get("teacher_vocab_size"):
        teacher_vocab_size = meta["teacher_vocab_size"]
    else:
        # Try loading teacher model config for accurate vocab_size
        from transformers import AutoConfig
        teacher_config = AutoConfig.from_pretrained(teacher_model_id or meta.get("model_id", student_id),
                                                     trust_remote_code=True)
        if hasattr(teacher_config, "vocab_size"):
            teacher_vocab_size = teacher_config.vocab_size

    print(f"Teacher vocab size: {teacher_vocab_size}")

    projector_path = cache_dir / "vocab_projector.pt"
    if projector_path.exists():
        print(f"Loading cached vocab projector: {projector_path}")
        saved = torch.load(projector_path, weights_only=True)
        teacher_to_student = saved["teacher_to_student"]
    else:
        print(f"Building vocab projection: teacher {teacher_vocab_size} → student {student_vocab_size}")
        teacher_to_student = build_vocab_projection(teacher_model_id or meta.get("model_id", student_id),
                                                     student_id,
                                                     device="cpu")
        # Pad to teacher_vocab_size (model.config.vocab_size may be > tokenizer.vocab_size)
        if teacher_to_student.size(0) < teacher_vocab_size:
            pad = torch.zeros(teacher_vocab_size - teacher_to_student.size(0), dtype=torch.long)
            teacher_to_student = torch.cat([teacher_to_student, pad])
        torch.save({"teacher_to_student": teacher_to_student, "teacher_vocab": teacher_vocab_size,
                     "student_vocab": student_vocab_size}, projector_path)
        print(f"Projection ready: {teacher_to_student.size(0)} teacher → student tokens")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    dataset = CachedLogitsDataset(cache_dir, max_length=max_length,
                                   student_tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Combine student model parameters (projector has no learnable params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if wandb:
        import trackio
        trackio.init(project=wandb_project, name=f"distill-{student_id.split('/')[-1]}",
                     config={"student": student_id, "teacher": teacher_model_id,
                             "temperature": temperature, "alpha": alpha,
                             "lr": learning_rate, "epochs": epochs})

    model.train()
    global_step = 0
    running_loss = 0.0

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logit_values = batch["logit_values"].to(device)     # [B, S, k] fp16
            logit_indices = batch["logit_indices"].to(device)   # [B, S, k] int32

            # Reconstruct teacher logits in TEACHER vocab space
            B, S, k = logit_values.shape
            teacher_logits = torch.full((B, S, teacher_vocab_size), float("-inf"),
                                        dtype=torch.float32, device=device)
            teacher_logits.scatter_(-1, logit_indices.long(), logit_values.float())

            # Student forward pass → logits in student vocab
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = outputs.logits  # [B, S, student_vocab]

            # Distillation loss: KL only over teacher's top-k positions
            # 1. Map teacher token indices → student token indices
            safe_idx = logit_indices.clamp(0, teacher_to_student.size(0) - 1)
            mapped_indices = teacher_to_student.to(device)[safe_idx]  # [B, S, k] in student vocab

            # 2. Extract student logits at those positions
            gathered_student = torch.gather(
                student_logits, -1, mapped_indices.long()
            )  # [B, S, k]

            # 3. Mask: exclude padded positions (logit_values == -inf)
            valid_mask = logit_values.float() > float("-inf")

            # 4. KL divergence over the k positions only
            T = temperature
            student_clean = gathered_student.float().masked_fill(~valid_mask, 0.0)
            teacher_clean = logit_values.float().masked_fill(~valid_mask, 0.0)

            # Debug: check for NaN sources
            if global_step == 0 and batch_idx == 0:
                n_valid = valid_mask.sum().item()
                print(f"  DEBUG: valid={n_valid}/{valid_mask.numel()}, "
                      f"student range=[{student_clean.min():.2f}, {student_clean.max():.2f}], "
                      f"teacher range=[{teacher_clean.min():.2f}, {teacher_clean.max():.2f}]")
                print(f"  DEBUG: mapped_indices range=[{mapped_indices.min()}, {mapped_indices.max()}], "
                      f"student_vocab={student_vocab_size}")

            student_log_probs = F.log_softmax(student_clean / T, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_clean / T, dim=-1)

            # Zero out invalid positions so they don't contribute to KL
            student_log_probs = student_log_probs.masked_fill(~valid_mask, 0.0)
            teacher_log_probs = teacher_log_probs.masked_fill(~valid_mask, 0.0)

            # KL divergence with masking
            loss_kl = F.kl_div(
                student_log_probs,
                teacher_log_probs,
                reduction="none",
                log_target=True,
            )  # [B, S, k]

            # Zero out invalid positions and average over valid ones
            loss_kl = loss_kl.masked_fill(~valid_mask, 0.0)
            valid_count = valid_mask.sum().clamp(min=1)
            loss_kl = loss_kl.sum() / valid_count * (T ** 2)

            # Hard label loss (cross-entropy on input_ids shifted by 1)
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
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
                        import trackio
                        trackio.log({
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

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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
        import trackio
        trackio.finish()

    print(f"Training complete. Model saved to {output_dir}")
    return output_dir
