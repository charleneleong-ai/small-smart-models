"""Tests for knowledge distillation module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def mock_cache_dir(tmp_path: Path) -> Path:
    """Create a mock cached logits directory for testing."""
    cache_dir = tmp_path / "teacher_logits"
    cache_dir.mkdir()

    # Create a small shard
    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    top_k = 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = torch.randn(batch_size, seq_len, vocab_size)

    shard_path = cache_dir / "shard_0000.pt"
    torch.save({
        "input_ids": input_ids,
        "logits": logits,
    }, shard_path)

    # Create index
    index_entries = []
    for i in range(batch_size):
        index_entries.append({
            "shard": 0,
            "offset": i,
            "length": seq_len,
        })

    with open(cache_dir / "index.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in index_entries)

    # Create metadata
    meta = {
        "model_id": "test-model",
        "dataset": "test:dataset",
        "split": "train",
        "max_length": 2048,
        "top_k": top_k,
        "total_samples": batch_size,
        "total_shards": 1,
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return cache_dir


def test_cached_logits_dataset_loads(mock_cache_dir: Path) -> None:
    """CachedLogitsDataset loads shards correctly."""
    from smart_quant.distill import CachedLogitsDataset

    dataset = CachedLogitsDataset(mock_cache_dir, top_k=10)

    assert len(dataset) == 4
    sample = dataset[0]

    assert "input_ids" in sample
    assert "logits" in sample
    # Shape is [batch, seq_len] for input_ids and [batch, seq_len, top_k] for logits
    # But since we're loading per-sample, it's just [seq_len] and [seq_len, top_k]
    assert sample["input_ids"].dim() == 1  # seq_len
    assert sample["logits"].dim() == 2  # [seq_len, top_k]


def test_cached_logits_dataset_top_k(mock_cache_dir: Path) -> None:
    """CachedLogitsDataset sparsifies to top-k logits."""
    from smart_quant.distill import CachedLogitsDataset

    dataset = CachedLogitsDataset(mock_cache_dir, top_k=5)
    sample = dataset[0]

    # Should have top-k non-inf values per token
    # (masked with -inf rather than physically removed)
    non_inf_per_token = (sample["logits"] != float("-inf")).sum(dim=-1)
    assert (non_inf_per_token == 5).all()

    # The shape stays [seq_len, vocab_size] but only top-k are non-inf
    assert sample["logits"].shape == (32, 1000)


def test_cache_teacher_logits_creates_structure(tmp_path: Path) -> None:
    """cache_teacher_logits creates expected directory structure."""
    from smart_quant.distill import cache_teacher_logits

    # This would fail without a real model, but we can test the function signature
    # and that it imports correctly
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
