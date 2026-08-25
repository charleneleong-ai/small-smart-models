"""Tests for knowledge distillation module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def mock_cache_dir(tmp_path: Path) -> Path:
    """Create a mock cached logits directory in sparse top-k format."""
    cache_dir = tmp_path / "teacher_logits"
    cache_dir.mkdir()

    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    top_k = 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logit_values = torch.randn(batch_size, seq_len, top_k)
    logit_indices = torch.randint(0, vocab_size, (batch_size, seq_len, top_k))

    shard_path = cache_dir / "shard_0000.pt"
    texts = [f"Sample text {i} for testing distillation pipeline" for i in range(batch_size)]
    torch.save({
        "input_ids": input_ids,
        "logit_values": logit_values.half(),
        "logit_indices": logit_indices.int(),
        "text": texts,
    }, shard_path)

    index_entries = []
    for i in range(batch_size):
        index_entries.append({
            "shard": 0,
            "offset": i,
            "length": seq_len,
        })

    with open(cache_dir / "index.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in index_entries)

    meta = {
        "model_id": "test-model",
        "dataset": "test:dataset",
        "split": "train",
        "max_length": 2048,
        "top_k": top_k,
        "format": "sparse_topk",
        "total_samples": batch_size,
        "total_shards": 1,
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return cache_dir


def test_cached_logits_dataset_loads(mock_cache_dir: Path) -> None:
    """CachedLogitsDataset loads sparse shards correctly."""
    from smart_quant.distill import CachedLogitsDataset

    dataset = CachedLogitsDataset(mock_cache_dir, max_length=32)

    assert len(dataset) == 4
    sample = dataset[0]

    assert "input_ids" in sample
    assert "logit_values" in sample
    assert "logit_indices" in sample
    assert sample["input_ids"].dim() == 1  # [seq_len]
    assert sample["logit_values"].dim() == 2  # [seq_len, k]
    assert sample["logit_indices"].dim() == 2  # [seq_len, k]


def test_cache_teacher_logits_importable() -> None:
    """cache_teacher_logits is importable and callable."""
    from smart_quant.distill import cache_teacher_logits
    assert callable(cache_teacher_logits)


def test_train_student_imports() -> None:
    """train_student function is importable and callable."""
    from smart_quant.distill import train_student
    assert callable(train_student)


def test_cli_commands_imports() -> None:
    """CLI commands are registered."""
    from smart_quant.cli import app

    commands = [cmd.name for cmd in app.registered_commands]
    assert "cache-teacher" in commands
    assert "distill" in commands
    assert "distill-eval" in commands


def test_training_config_save(tmp_path: Path) -> None:
    """Training config is saved correctly."""
    config = {
        "student_id": "test-student",
        "cache_dir": str(tmp_path),
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "temperature": 4.0,
        "alpha": 0.7,
    }

    config_path = tmp_path / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    with open(config_path) as f:
        loaded = json.load(f)

    assert loaded["student_id"] == "test-student"
    assert loaded["temperature"] == 4.0


def test_sparse_shard_reconstruction(mock_cache_dir: Path) -> None:
    """Sparse logits can be reconstructed into dense form for KL computation."""
    from smart_quant.distill import CachedLogitsDataset

    dataset = CachedLogitsDataset(mock_cache_dir, max_length=32)
    sample = dataset[0]

    seq_len = 32
    vocab_size = 1000
    logit_values = sample["logit_values"]    # [seq_len, k]
    logit_indices = sample["logit_indices"]  # [seq_len, k]

    # Reconstruct dense
    dense = torch.full((seq_len, vocab_size), float("-inf"))
    dense.scatter_(-1, logit_indices.long(), logit_values.float())

    # Check: positions with top-k values should be non-inf
    non_inf = (dense != float("-inf")).sum(dim=-1)
    # May be less than k due to duplicate indices, but should be > 0
    assert (non_inf > 0).all()
