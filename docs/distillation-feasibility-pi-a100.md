# Distillation Feasibility on pi-a100-80gb

> **For future Claude:** This is a practical guide for what we can actually do on our single A100-80GB box, focusing on distillation as the highest-impact technique.

## Hardware Profile

- **GPU:** A100-80GB (single)
- **Usable VRAM:** ~70GB after model loading
- **Storage:** Local NVMe (fast I/O)
- **Network:** SSH access for remote execution

## Feasibility Matrix

### ✅ FEASIBLE (Can do on single A100)

| Task | Model Size | VRAM Needed | Time Estimate | Impact |
|------|-----------|-------------|---------------|--------|
| Cache teacher logits | 35B (fp16) | ~70GB | 2-4 GPU-hrs | High |
| Train 1.5B student | 1.5B | ~12GB | 8-16 GPU-hrs | High |
| Train 3B student | 3B | ~24GB | 16-32 GPU-hrs | High |
| QAD quantization | 1.5B-3B | ~12-24GB | 4-8 GPU-hrs | Medium |
| Generate synthetic data | 35B (inference) | ~70GB | 2-4 GPU-hrs | Medium |
| Evaluate models | Any | ~70GB max | 2-4 GPU-hrs | Low |

### ⚠️ PARTIALLY FEASIBLE (With optimizations)

| Task | Model Size | VRAM Needed | Time Estimate | Constraints |
|------|-----------|-------------|---------------|-------------|
| Train 7B student | 7B | ~48GB | 32-64 GPU-hrs | Gradient checkpointing required |
| Multi-step distillation | 35B→7B→1.5B | ~70GB | 48-96 GPU-hrs | Sequential, not parallel |
| Mamba hybrid training | 1.5B-3B | ~12-24GB | 16-32 GPU-hrs | Custom architecture needed |

### ❌ NOT FEASIBLE (Need multi-GPU or more memory)

| Task | Why Not | Alternative |
|------|---------|-------------|
| Pretrain from scratch | Need 100s of GPU-days | Use existing base models |
| Large-scale RLHF | Need distributed training | Use offline RL on cached data |
| Model merging | Need multiple fine-tuned models | Merge after individual training |
| Neural architecture search | Need many parallel experiments | Manual architecture choices |

## Implementation Roadmap

### Phase 1: Cache Teacher Logits (Day 1)

**Goal:** Extract knowledge from Qwen3.6-35B-A3B teacher

```bash
# 1. Load teacher model
# 2. Run over calibration set (512 C4 rows)
# 3. Cache top-100 logits per token
# 4. Store to disk for student training
```

**Output:** `experiments/teacher_logits/` (~50GB cached logits)

**Time:** 2-4 GPU-hours

### Phase 2: Train 1.5B Student (Days 2-3)

**Goal:** Distill teacher knowledge into small model

```bash
# 1. Load cached teacher logits
# 2. Initialize Qwen2.5-1.5B as student
# 3. Train against cached logits (10k steps)
# 4. Evaluate on capability battery
```

**Output:** `experiments/distilled-models/student-1.5b/`

**Time:** 8-16 GPU-hours

### Phase 3: QAD Quantization (Day 4)

**Goal:** Quantize student to NVFP4 while recovering accuracy

```bash
# 1. Load distilled student
# 2. Apply QAD with teacher logits
# 3. Quantize to NVFP4
# 4. Evaluate accuracy recovery
```

**Output:** `experiments/qad-models/student-1.5b-nvfp4/`

**Time:** 4-8 GPU-hours

### Phase 4: Evaluation (Day 5)

**Goal:** Measure distillation quality

```bash
# 1. Run capability battery on all models
# 2. Compare: fp16 teacher vs distilled vs quantized
# 3. Generate quality-vs-bpw plot
# 4. Write up findings
```

**Output:** Updated `docs/experiments/bits-per-brain.md` with distillation results

**Time:** 2-4 GPU-hours

## Total Investment

| Phase | GPU-hours | Wall-clock (sequential) |
|-------|-----------|------------------------|
| Phase 1: Cache logits | 3 | 3 hours |
| Phase 2: Train student | 12 | 12 hours |
| Phase 3: QAD | 6 | 6 hours |
| Phase 4: Evaluate | 3 | 3 hours |
| **Total** | **24** | **24 hours** |

## Expected Outcomes

### Best Case
- 1.5B student achieves 80%+ of teacher's capability
- NVFP4 quantization recovers to 95%+ of student's fp16 accuracy
- Total model size: ~1GB (vs 70GB for teacher)
- Inference speed: 10-20x faster than teacher

### Realistic Case
- 1.5B student achieves 60-70% of teacher's capability
- NVFP4 quantization recovers to 90%+ of student's fp16 accuracy
- Total model size: ~1GB
- Inference speed: 10-20x faster than teacher

### Worst Case
- Distillation fails to transfer knowledge effectively
- QAD degrades accuracy significantly
- Need to try different student architectures or training recipes

## Key Risks

1. **Teacher logits too large to cache**
   - Mitigation: Use top-K sparsification (keep top-100 logits only)

2. **Student training unstable**
   - Mitigation: Lower learning rate, gradient clipping, warmup

3. **QAD degrades accuracy**
   - Mitigation: Try different quantization levels (INT8, INT4)

4. **Evaluation shows poor transfer**
   - Mitigation: Try different student architectures (Mamba, MoE)

## Success Criteria

The distillation project succeeds if:

1. **Student quality:** 1.5B student achieves >50% of teacher's MMLU/GSM8K scores
2. **Quantization recovery:** NVFP4 student achieves >90% of fp16 student's accuracy
3. **Efficiency gain:** Total model size <2GB, inference speed >10x faster than teacher
4. **Novel finding:** Distillation beats direct quantization of teacher model

## Next Steps After Phase 4

If distillation works well:

1. **Multi-step distillation:** 35B → 7B → 1.5B progressive compression
2. **Architecture experiments:** Mamba hybrid, Per-Layer Embeddings
3. **Domain specialization:** Distill for specific tasks (math, code, reasoning)
4. **Scale up:** Try 7B student if 1.5B works well

If distillation fails:

1. **Analyze failure modes:** Where does knowledge transfer break down?
2. **Try different teachers:** Qwen3.5-9B, Gemma 3 4B
3. **Try different students:** Phi-4-mini architecture, Gemma 3n
4. **Focus on data curation:** Synthetic reasoning data instead of distillation

---

*Last updated: 2026-08-23*
*Status: Implementation plan*
*Owner: pi-a100-80gb*
