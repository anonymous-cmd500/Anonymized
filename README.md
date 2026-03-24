# EEG-to-Text: Neural Decoding of Language from Brain Signals

> **S4D Encoder + BART Decoder** — Translating word-level EEG recordings into natural language text using Structured State Spaces and pretrained language models.
**Dataset in pickle format is available in https://drive.google.com/drive/folders/1TWMDhZFfOhglPuUnT2YZMpEHe3T_3HPu?usp=sharing**
---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Training Pipeline](#training-pipeline)
5. [Loss Functions](#loss-functions)
6. [Decoding Strategies](#decoding-strategies)
7. [Key Novelties](#key-novelties)
8. [Results](#results)
9. [Project Structure](#project-structure)
10. [Usage](#usage)
11. [Requirements](#requirements)

---

## Overview

This project implements an end-to-end neural decoder that converts **word-level EEG (electroencephalography) signals** recorded while subjects read sentences into coherent **English text**. The system reads brain activity captured from 105 EEG channels across 8 frequency bands and generates the corresponding sentence that was being read.

**Core approach:** A custom **S4D (Diagonal State Space Model) encoder** processes sequential EEG word representations, and its output is wired as the **encoder hidden states** into a pretrained **BART decoder** via cross-attention. The decoder generates text token-by-token, conditioned solely on the brain signals.

```
 ╔════════════════════════╗          ╔══════════════════════════╗
 ║ Word-level EEG         ║          ║ Target sentence tokens   ║
 ║ (B, L, 840)            ║          ║ (decoder_input_ids)      ║
 ╚═══════╤════════════════╝          ╚══════════╤═══════════════╝
         │                                      │
    S4D Encoder (6L)                       BART Tokenizer
    + Attention Gate                            │
         │                                      │
 encoder_hidden_states ──► BART Decoder (cross-attention) ──► text tokens
 (B, L, 768)               (self-attn + cross-attn + FFN)
```

---

## Architecture

### 1. S4D EEG Encoder

The encoder is built from scratch using the **Diagonal State Space Model (S4D)** — a structured sequence model that excels at capturing long-range dependencies in sequential data.

| Component | Details |
|---|---|
| **Input dimension** | 840 (105 channels × 8 frequency bands) |
| **S4D dimension** | 512 |
| **Number of layers** | 6 |
| **State dimension (N)** | 64 |
| **Bidirectional** | Yes (processes EEG left-to-right and right-to-left) |
| **Initialisation** | HiPPO-LegS (optimal for long-range memory) |
| **Discretisation** | Bilinear (Tustin) transform |
| **Output projection** | MLP → 768 (BART-base hidden dim) |

**Why S4D over Transformers/LSTMs?**
- S4D has **linear complexity** in sequence length (vs quadratic for Transformers)
- HiPPO initialisation provides principled **long-range memory** — critical for capturing temporal EEG dynamics across words in a sentence
- Bidirectional processing captures both forward and backward context of the full sentence EEG
- The diagonal parameterisation (S4**D**) is computationally efficient while retaining the expressiveness of the full S4 model

**S4D kernel computation:**
The S4D kernel is implemented from the closed-form diagonal SSM. For each layer:
1. Continuous parameters `A` (diagonal complex, Hurwitz-stable), `B`, `C`, `D` are maintained
2. Discretised via the bilinear transform: `Ā = (I + Δt/2 · A)⁻¹ · (I - Δt/2 · A)`
3. The convolution kernel `K` of length `L` is computed: `K_k = C · Ā^k · B̄`
4. Convolution is performed in the frequency domain via FFT for efficiency

Each S4D layer includes: **S4DKernel → Dropout → GLU activation → LayerNorm → Residual connection**

### 2. EEG Attention Gate

A learned gating mechanism applied to encoder hidden states **before** they enter BART's cross-attention:

```
gate = σ(W · h + bias)
output = gate ⊙ h + (1 - gate) ⊙ null_vector
```

- **Bias initialised to +1.0** → gate starts ~73% open, giving the decoder real EEG signal from step one
- **Learnable null vector** provides a gradient-friendly baseline when the gate is closed
- Prevents dead-start scenarios where cross-attention receives zero signal

### 3. BART Decoder

Uses the **facebook/bart-base** pretrained decoder (6 layers, 768d, 6 attention heads):

- **Cross-attention layers** attend over S4D encoder outputs (EEG hidden states)
- **Self-attention dampening (scale=0.0):** Forward hooks zero out self-attention outputs, forcing the decoder to rely entirely on cross-attention (EEG signal) rather than its own language model prior for content generation
- **Gradient checkpointing** enabled to reduce VRAM usage

### 4. Contrastive Projection Heads (CLIP-style)

Separate MLP projection heads for EEG and text embeddings:
```
eeg_proj:  Linear(768→768) → GELU → Linear(768→256)
text_proj: Linear(768→768) → GELU → Linear(768→256)
```
These project mean-pooled representations into a 256-dim space for InfoNCE alignment, decoupling the contrastive objective from the per-token cross-attention representations.

### 5. EEG Vocabulary Prior Head

Predicts a bag-of-words distribution from mean-pooled EEG using BART's shared embedding matrix:
```
mean_pool(EEG) → Linear(768→768) → GELU → BART_embeddings^T → vocab logits
```
Trained with binary cross-entropy on which tokens appear in the target sentence.

---

## Dataset

### ZuCo 1.0 (Zurich Cognitive Language Processing Corpus)

| Property | Value |
|---|---|
| **Source** | ZuCo v1.0 only |
| **Tasks** | Task 1: Sentiment Reading (SR), Task 2: Normal Reading (NR), Task 3: Task-Specific Reading (TSR) |
| **Subjects** | 12 (ZAB, ZDM, ZDN, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW, ZMG, ZPH) |
| **EEG device** | 128-channel EEG (105 usable channels after artefact rejection) |
| **Feature level** | **Word-level** — EEG features are extracted per word fixation, not per sentence |
| **Frequency bands** | 8 bands: θ₁, θ₂, α₁, α₂, β₁, β₂, γ₁, γ₂ |
| **Feature dimension** | 105 channels × 8 bands = **840 features per word** |
| **Feature type** | GD (Gaze Duration) — EEG averaged over the entire first-pass fixation of each word |
| **Total usable samples** | 12,579 (word-level EEG matrix, sentence) pairs |
| **Unique sentences** | 1,039 |
| **Max words/sentence** | 56 (padded) |

### Data Split (sentence-disjoint, primary)

The 1,039 unique sentences are partitioned into non-overlapping train/dev/test groups. All EEG recordings of each sentence (from all subjects) are assigned to the same split. **Zero text overlap** between partitions.

| Split | Unique Sentences | Samples (all subjects) | Sentence Overlap |
|---|---|---|---|
| **Train** | 831 (80%) | 9,978 | — |
| **Dev** | 103 (10%) | 1,285 | 0 with Train, 0 with Test |
| **Test** | 105 (10%) | 1,316 | 0 with Train, 0 with Dev |
| **Total** | 1,039 | 12,579 | **Zero overlap verified** |

### Data Split 2: Subject wise 
We evaluate the model under a strict
subject-independent setting. Training was performed on EEG
recordings from Train subjects: 9, Dev subject: 1 and Test
subjects: 2. This setting ensures that the model must general
ize across participants rather than exploiting subject-specific
neural patterns

**Subjects in Each Split**
```
Train: ['ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZPH']
Dev:   ['ZAB']
Test:  ['ZKW', 'ZMG']
```

**Final Counts**
| Split | Samples |
|------|--------|
| Train | 9300 |
| Dev | 1090 |
| Test | 2189 |
| **Total** | **12579** |

### Preprocessing

- **Per-channel z-score normalisation:** Computed on training split only, applied to all splits
- **EEG augmentation (training only):** Gaussian noise (σ=0.1), random channel dropout (5%), temporal shifting (±1 word)

---

## Training Pipeline

Training uses a **two-phase curriculum** designed to progressively build cross-modal alignment:

### Phase 1: EEG Encoder Warm-Up (Epochs 1–20)

| Setting | Value |
|---|---|
| **Frozen** | Entire BART (all parameters) |
| **Trainable** | S4D encoder, MLP projection, attention gate, projection heads, vocab prior head |
| **Learning rate** | 1×10⁻⁴ |
| **Batch size** | 32 (×4 accumulation = 128 effective) |
| **Loss** | L_lm + 2.0 · L_InfoNCE + 1.0 · L_vocab |
| **Self-attn scale** | 1.0 (no dampening) |

**Rationale:** Before cross-attention can work, the EEG encoder must produce vectors that live in BART's embedding space. Phase 1 trains the S4D encoder to output representations that align with BART's text encoder via the InfoNCE loss.

### Phase 2: Cross-Attention Fine-Tuning (Epochs 21–60)

| Setting | Value |
|---|---|
| **Frozen** | BART text encoder, BART decoder self-attention, embedding layers |
| **Trainable** | BART cross-attention layers, FFN layers, LayerNorms, S4D encoder, gate |
| **Learning rate** | 3×10⁻⁵ |
| **Batch size** | 16 (×8 accumulation = 128 effective) |
| **Loss** | L_lm + 2.0 · L_InfoNCE + 0.1 · L_attention_entropy + 1.0 · L_vocab |
| **Self-attn scale** | 0.0 (fully EEG-dependent) |
| **Word dropout** | Progressive 30% → 75% (forces cross-attention reliance) |

**Rationale:** With aligned EEG representations, Phase 2 fine-tunes BART's cross-attention to attend to EEG positions. Self-attention dampening and progressive word dropout force the decoder to extract content from EEG rather than relying on its language model.

### Shared Settings

| Setting | Value |
|---|---|
| **Optimiser** | AdamW (weight decay 0.01) |
| **LR schedule** | Cosine annealing with 500-step linear warmup |
| **Gradient clipping** | Max norm 1.0 |
| **Mixed precision** | FP16 (autocast + GradScaler) |
| **Checkpoint selection** | Best validation BERTScore F1 |
| **Hardware** | Single GPU (8GB+ VRAM) |

---

## Loss Functions

The total training loss combines four complementary objectives:

**L = L_lm + λ_c · L_InfoNCE + λ_a · L_attn + λ_v · L_vocab**

### 1. Label-Smoothed Cross-Entropy (L_lm)

Standard language modelling loss with label smoothing (ε=0.1):

**L_lm = (1-ε) · NLL + ε · uniform**

Prevents overconfident predictions and regularises the output distribution.

### 2. InfoNCE Contrastive Alignment (L_InfoNCE, λ=2.0)

**CLIP-style symmetric InfoNCE** that aligns EEG and text representations in a shared embedding space. Computes cosine similarity between all EEG-text pairs in a batch and applies cross-entropy with temperature τ=0.07.

**This is key for cross-modal alignment:** Without it, the S4D encoder output lives in a random direction relative to BART's representational space, and cross-attention receives unstructured features that yield unfocused attention distributions.

### 3. Cross-Attention Entropy Regularisation (L_attn, λ=0.1)

Penalises high entropy (uniform distribution) in cross-attention weights. Encourages the decoder to form **peaked, focused** cross-attention patterns over specific EEG positions rather than attending uniformly.

### 4. EEG Vocabulary Prior (L_vocab, λ=1.0)

Binary cross-entropy for bag-of-words prediction from mean-pooled EEG. Trains the EEG encoder to predict which vocabulary tokens appear in the target sentence, providing an explicit content signal.

---

## Decoding Strategies

We evaluate four generation strategies to understand model behaviour:

### 1. Greedy Decoding (PRIMARY)

```
At each step: y_t = argmax P(y | y_{<t}, EEG)
```

- **Deterministic** — same EEG always produces the same output
- **num_beams=1**, no sampling
- **Best performing** strategy for this task (BERTScore F1 = 0.709)
- Indicates the model's learned EEG→text mapping is maximal at the mode of the distribution


### 2. Teacher-Forced Argmax (UPPER BOUND)

```
Decoder receives actual reference tokens as input; output = argmax of logits
```

- **Not a real decoding strategy** — the model "cheats" by seeing reference tokens
- Represents the **theoretical upper bound** of what BART can reconstruct given perfect input context

---

## Key Novelties

### 1. S4D as EEG Sequence Encoder

**First application of Structured State Space Models (S4D) for EEG-to-text decoding.** Prior work uses LSTMs or Transformers for EEG encoding. S4D's HiPPO-LegS initialisation provides principled long-range memory, and the bidirectional configuration captures full-sentence temporal context from EEG — achieving this with linear computational complexity.

### 2. Cross-Modal Alignment via InfoNCE Before Fine-Tuning

A **two-phase curriculum** where the EEG encoder must first learn to produce representations aligned with BART's text embedding space (Phase 1: InfoNCE alignment) before cross-attention fine-tuning begins (Phase 2). This prevents the decoder from ignoring the EEG encoder and relying purely on its language model prior.

### 3. Self-Attention Dampening

A novel **forward-hook-based mechanism** that scales BART decoder self-attention outputs by 0.0 during Phase 2 training and inference (completely suppressing the language model prior). This forces the decoder to rely on **cross-attention (EEG signal) for content** rather than falling back on its pretrained language model, preventing mode collapse where all inputs produce generic text.

### 4. Progressive Word Dropout for Cross-Attention Forcing

During Phase 2, decoder input word dropout is progressively increased from 30% → 75% across training. Higher dropout rates corrupt the decoder's self-attention input, forcing it to extract content through cross-attention from EEG representations.

### 5. EEG Vocabulary Prior with Shared Embeddings

An auxiliary **bag-of-words prediction head** that uses BART's shared embedding matrix to predict which tokens appear in the target sentence from mean-pooled EEG. This provides an explicit content signal during training and can optionally supply a vocabulary prior during beam search generation.

### 6. Attention Entropy Regularisation

An explicit loss term that penalises uniform (high-entropy) cross-attention distributions, encouraging the decoder to form **peaked, focused attention patterns** over specific EEG word positions — ensuring the decoder selectively attends to EEG features rather than spreading attention uniformly.

### 7. Multi-Strategy Decoding Evaluation Framework

A comprehensive evaluation framework that runs **four parallel decoding strategies** (greedy, beam, nucleus, teacher-forced) and computes **per-sample BERTScore** with full ranking, enabling fine-grained analysis of where and why the model succeeds or fails.

---

## Results

### Primary Results — Sentence-Disjoint Split (1,316 test samples, 105 held-out sentences)

Best checkpoint at epoch 43 (trained 54 epochs), validation BERTScore F1 = 0.691.

| Metric | Greedy | TF (upper bound) |
|---|---|---|
| **BERTScore F1** | **0.709** | 0.754 |
| **BLEU-1** | 0.224 | 0.397 |
| **BLEU-2** | 0.089 | 0.233 |
| **BLEU-3** | 0.045 | 0.148 |
| **BLEU-4** | 0.025 | 0.099 |
| **ROUGE-1 P** | 0.197 | 0.348 |
| **ROUGE-1 R** | 0.190 | 0.338 |
| **ROUGE-1 F** | 0.189 | 0.342 |
| **ROUGE-2 P** | 0.034 | 0.112 |
| **ROUGE-2 R** | 0.032 | 0.110 |
| **ROUGE-2 F** | 0.032 | 0.111 |
| **ROUGE-L P** | 0.155 | 0.312 |
| **ROUGE-L R** | 0.149 | 0.303 |
| **ROUGE-L F** | 0.148 | 0.307 |
| **WER** | 1.017 | 0.759 |

### BERTScore F1 Distribution (Greedy)

| Statistic | Value |
|---|---|
| P10 | 0.652 |
| P25 | 0.676 |
| P50 (Median) | 0.707 |
| P75 | 0.742 |
| P90 | 0.768 |
| Min / Max | 0.537 / 0.902 |

---

## Project Structure

### Core Package (`eeg_to_text/`)

```
eeg_to_text/
├── __init__.py
├── config.py                    # All hyperparameters (dataclass)
├── train.py                     # Training entry point
├── evaluate.py                  # Evaluation entry point
│
├── models/
│   ├── s4d_encoder.py           # S4D kernel + S4D EEG encoder (from scratch)
│   ├── attention_gate.py        # Learned gating for encoder outputs
│   └── eeg_to_text.py           # Full model: S4D + Gate + BART wiring
│
├── data/
│   ├── preprocessing.py         # Pickle loading, feature extraction, normalisation
│   └── dataset.py               # PyTorch Dataset, collate, train/dev/test split
│
├── training/
│   ├── trainer.py               # Two-phase curriculum trainer
│   ├── losses.py                # All 5 loss functions + composite loss
│   └── scheduler.py             # Cosine LR scheduler with warmup
│
└── evaluation/
    └── metrics.py               # BLEU, ROUGE, BERTScore, WER + multi-mode eval
```

### Evaluation & Results

```
full_eval.py                            # Generates all predictions (4 modes) + metrics

Results/
├── all_predictions_ranked.txt          # All test predictions (4 modes) ranked
├── all_predictions_ranked.csv          # Same in CSV format
├── final_metrics.txt                   # Aggregate scores table
└── checkpoints/
    └── best.pt                         # Best model checkpoint (epoch 36)
```

### Data

```
dataset/
├── task1-SR-dataset.pickle             # ZuCo v1.0 Task 1: Sentiment Reading
├── task2-NR-dataset.pickle             # ZuCo v1.0 Task 2: Normal Reading
└── task3-TSR-dataset.pickle            # ZuCo v1.0 Task 3: Task-Specific Reading
```

---

## Usage

### Training

```bash
# Full two-phase training (Phase 1: 20 epochs + Phase 2: 40 epochs)
python -m eeg_to_text.train --fp16 --device cuda

# Phase 2 only (resume from Phase 1 checkpoint)
python -m eeg_to_text.train --fp16 --device cuda --phase2_only
```

### Evaluation

```bash
# Full evaluation with all decoding modes + per-sample predictions
python full_eval.py

# Evaluate a specific checkpoint
python -m eeg_to_text.evaluate --checkpoint Results/checkpoints/best.pt
```

### Configuration

All hyperparameters are centralised in `eeg_to_text/config.py` as a Python dataclass. Key parameters:

```python
# Model
s4d_dim = 512           # S4D internal dimension
s4d_layers = 6          # Number of S4D blocks
s4d_state_dim = 64      # State space dimension N
self_attn_scale = 0.0   # Self-attention dampening factor (0=fully EEG-dependent)

# Training
phase1_epochs = 20      # EEG encoder warm-up
phase2_epochs = 40      # Cross-attention fine-tuning
lambda_contrastive = 2.0  # InfoNCE weight
label_smoothing = 0.1

# Generation
repetition_penalty = 1.3
max_gen_length = 56
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- transformers (Hugging Face)
- bert-score
- rouge-score
- jiwer (for WER)
- numpy, tqdm

```bash
pip install -r eeg_to_text/requirements.txt
```

---


---

## Acknowledgements

- **ZuCo Dataset:** Hollenstein et al. (2018, 2020) — [ZuCo: A Simultaneous EEG and Eye-tracking Resource for Natural Sentence Reading](https://www.nature.com/articles/sdata2018291)
- **S4D:** Gu et al. (2022) — [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893)
- **BART:** Lewis et al. (2020) — [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- **InfoNCE / CLIP:** Radford et al. (2021) — [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
