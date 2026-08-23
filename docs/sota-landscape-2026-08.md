# Small Language Model Landscape — August 2026

> **For future Claude:** This is a research snapshot of the SLM field as of 2026-08-23, capturing what techniques are actually working to make small models smart, and what we can feasibly do on pi-a100-80gb. The bits-per-brain study focused on post-training compression (quantization); this doc captures why the field has moved to training-time techniques instead.

## The Shift: Compression → Training

Our bits-per-brain study spent 9 phases exploring quantization techniques (PQ, residual VQ, weighted PQ, GPTQ compensation, E8 lattice). The finding: uniform shared-codebook PQ is a hard local optimum at 2.0-2.5 bpw.

**But the field has moved on.** The techniques actually enabling smart small models in 2026 are:

1. **Knowledge distillation** (not quantization) as the primary compression
2. **Data curation** (not weight precision) as the key quality lever
3. **Architecture innovation** (not post-hoc optimization)
4. **Multi-step training pipelines** (SFT → RL → model merging)

## Top Performing Small Models (Sub-10B)

| Model | Params | Key Benchmarks | Key Technique |
|-------|--------|----------------|---------------|
| **Qwen3.5-9B** | 9B | 82.5 MMLU-Pro, 81.7 GPQA Diamond | Hybrid Gated DeltaNet architecture |
| **Phi-4-mini** | 3.8B | 67.3% MMLU, 88.6% GSM8K, 82.3% HumanEval | Synthetic "textbook" data training |
| **Gemma 3 4B** | 4B | 89.2% GSM8K, 71.3% HumanEval | Distillation + multimodal |
| **SmolLM3-3B** | 3B | 44.1% MMLU, 67.6% GSM8K | Fully open training pipeline |
| **Qwen3-8B** | 8B | Strong code generation | Apache 2.0, 32K context |
| **Llama 3.2 3B** | 3B | 63.4% MMLU, 77.7% GSM8K | Widely deployed, ecosystem |

## The 6 Techniques Enabling Smart Small Models

### 1. Knowledge Distillation (Most Important)

**What it is:** Train a smaller student model to mimic a larger teacher's behavior on soft probability distributions, not just hard labels.

**Key results:**
- **Phi-4-mini**: Distilled from larger teachers on synthetic "textbook" data
- **Gemma 3**: Knowledge distillation from larger Gemma models
- **Apertus Mini**: 16 models distilled from 8B teacher, 10x faster than pretraining
- **DeepSeek-R1-Distill**: 1.5B model outperforms GPT-4o on AIME (28.9%)

**Offline distillation breakthrough:**
- Cache teacher's top-K logits once, train student against cache
- 29% faster per iteration, 41% higher throughput
- Matches online quality while removing teacher from memory
- Enables training at 32,768 tokens context on single GPU

**Multi-step distillation:**
- Progressive compression through intermediate-sized models
- Consistently improves ROUGE-L and perplexity over single-step
- Avoids the "capacity cliff" when jumping directly from large to small

### 2. Data Curation (The Secret Sauce)

**Phi's approach:** "Textbook-quality" synthetic data beats massive raw web scrapes

- Training on high-quality synthetic reasoning data rather than quantity
- Microsoft's recipe: mixing high-quality synthetic data with filtered web content
- The training data quality matters more than model size

### 3. Quantization-Aware Distillation (QAD)

**NVIDIA's method for NVFP4 inference accuracy recovery:**
- Distills full-precision teacher into quantized student
- Robust to data quality, recovers near-BF16 accuracy
- Better than traditional QAT for RL-trained models
- Practical for multi-stage post-training pipelines (SFT → RL → merging)

### 4. Architecture Innovation

**Mamba/SSM hybrids:**
- Linear-time complexity, no KV cache
- Best for streaming/long-context applications
- Jamba Reasoning 3B: Mamba/Transformer hybrid, under 2GB at Q4

**Per-Layer Embeddings (Gemma 3n):**
- 8B raw parameters but 3B memory footprint
- First sub-10B model above 1300 LMArena Elo
- Purpose-built for mobile deployment

**Gated DeltaNet:**
- Hybrid attention (local sliding window + global)
- Qwen3.5 series uses this architecture

**MoE with small active parameters:**
- ZAYA1: 8B total, 0.76B active
- LFM2.5-8B-A1B: 8.3B total, 1.5B active
- Qwen3.6-35B-A3B: 35B total, 3B active (our target model)

### 5. Quantization (Mature but Limited Upside)

**4-bit quantization (GPTQ/AWQ) is the sweet spot:**
- 71% memory reduction
- 83% throughput increase
- <2% accuracy drop on general tasks
- ~4% drop on complex reasoning

**But it's a "scalpel, not a sword"** — quantization compresses; distillation creates capability.

### 6. Reinforcement Learning from Human Feedback (RLHF)

**DeepSeek-R1 approach:**
- Large-scale RL applied directly to base models
- No SFT as preliminary step
- Distilled into smaller models (1.5B-70B)
- 1.5B model outperforms GPT-4o on math benchmarks

## What's Feasible on pi-a100-80gb

### Hardware Constraints
- **Single A100-80GB GPU**
- **70GB usable** after model loading (fp16 Qwen3.6-35B-A3B uses ~70GB)
- **Sequential execution** — only one fp16 model fits at a time

### What We CAN Do

#### 1. Distillation (Highest Impact)

**Feasible approach:**
- Load Qwen3.6-35B-A3B as teacher (fp16, ~70GB)
- Cache teacher logits on calibration set (C4, 512 rows)
- Train smaller student (1.5B-3B) against cached logits
- Use offline distillation to avoid memory issues

**Implementation:**
```bash
# Step 1: Cache teacher logits
python cache_teacher_logits.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset allenai/c4 \
  --output-dir experiments/teacher_logits/ \
  --calib-rows 512

# Step 2: Train student against cache
python train_student.py \
  --teacher-cache experiments/teacher_logits/ \
  --student-config configs/student-1.5b.yaml \
  --output-dir experiments/distilled-models/
```

**Estimated time:** 2-4 GPU-hours for caching, 8-16 GPU-hours for training

#### 2. Quantization-Aware Distillation

**Feasible approach:**
- Take distilled student model
- Apply QAD to recover NVFP4 accuracy
- Use NVIDIA's chunked KL loss for efficiency

**Implementation:**
```bash
# QAD on distilled model
python qad_train.py \
  --teacher-cache experiments/teacher_logits/ \
  --student experiments/distilled-models/student-1.5b \
  --quantize-to nvfp4 \
  --output-dir experiments/qad-models/
```

**Estimated time:** 4-8 GPU-hours

#### 3. Architecture Experiments

**Feasible approaches:**
- Train Mamba/SSM hybrid student
- Experiment with Per-Layer Embeddings
- Test MoE with small active parameters

**Constraints:**
- Limited to models that fit in 80GB during training
- Can train 1.5B-3B models comfortably
- 7B models possible with gradient checkpointing

#### 4. Data Curation Experiments

**Feasible approaches:**
- Generate synthetic "textbook" data from teacher
- Curate high-quality reasoning chains
- Create domain-specific training sets

**Implementation:**
```bash
# Generate synthetic data from teacher
python generate_synthetic_data.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --prompts prompts/reasoning.jsonl \
  --output experiments/synthetic_data.jsonl \
  --num-samples 10000
```

**Estimated time:** 2-4 GPU-hours

### What We CANNOT Do

#### 1. Train Large Models from Scratch
- **Cannot train 7B+ models** from scratch on single A100
- **Cannot pretrain** — requires multiple GPUs and weeks of compute
- **Cannot do large-scale RL** — needs distributed training

#### 2. Multi-Stage Post-Training Pipelines
- **Cannot replicate full SFT → RL → merging** pipelines
- **Cannot do model merging** of multiple fine-tuned models
- **Cannot do large-scale RLHF** — needs distributed training

#### 3. Architecture Search
- **Cannot do neural architecture search** — requires many parallel experiments
- **Cannot train multiple large models** simultaneously

## Recommended Next Steps

### Immediate (This Week)

1. **Cache teacher logits** from Qwen3.6-35B-A3B
   - Use 512 C4 rows (matching our calibration set)
   - Store top-100 logits per token
   - Estimated: 2 GPU-hours

2. **Train 1.5B student** against cached logits
   - Use Qwen2.5-1.5B as architecture base
   - Train on cached logits for 10k steps
   - Estimated: 8 GPU-hours

3. **Evaluate distilled model**
   - Run capability battery (arc, hellaswag, winogrande, gsm8k, mmlu)
   - Compare to fp16 teacher and quantized baselines
   - Estimated: 2 GPU-hours

### Short-Term (Next 2 Weeks)

4. **Apply QAD to distilled model**
   - Quantize to NVFP4
   - Recover accuracy with distillation
   - Estimated: 4 GPU-hours

5. **Experiment with Mamba hybrid**
   - Train Mamba/Transformer hybrid student
   - Compare to pure Transformer student
   - Estimated: 8 GPU-hours

6. **Generate synthetic reasoning data**
   - Use teacher to create chain-of-thought data
   - Train student on reasoning tasks
   - Estimated: 4 GPU-hours

### Medium-Term (Next Month)

7. **Multi-step distillation**
   - Distill 35B → 7B → 1.5B progressively
   - Compare to direct 35B → 1.5B
   - Estimated: 24 GPU-hours

8. **Architecture experiments**
   - Test Per-Layer Embeddings
   - Experiment with MoE routing
   - Estimated: 16 GPU-hours

## Budget Estimate

| Task | GPU-hours | Cost (A100 $2/hr) |
|------|-----------|-------------------|
| Cache teacher logits | 2 | $4 |
| Train 1.5B student | 8 | $16 |
| Evaluate | 2 | $4 |
| QAD | 4 | $8 |
| Mamba hybrid | 8 | $16 |
| Synthetic data | 4 | $8 |
| Multi-step distillation | 24 | $48 |
| Architecture experiments | 16 | $32 |
| **Total** | **68** | **$136** |

## Key Insight

**Our bits-per-brain study was asking the wrong question.** We asked "how few bits can we use?" when we should have asked "how do we train small models that are genuinely smart?"

The answer in 2026 is clear:
1. **Distill from large teachers** (not compress large models)
2. **Curate high-quality data** (not optimize weight precision)
3. **Use architecture innovation** (not post-hoc optimization)
4. **Apply multi-stage training** (not single-step compression)

The 9 phases of our quantization study taught us that uniform PQ is a hard local optimum. Now we know why: **the field moved to a fundamentally different approach** — training better small models from scratch, not compressing large ones.

---

*Last updated: 2026-08-23*
*Status: Research snapshot*
*Next review: When new model releases or technique breakthroughs occur*
