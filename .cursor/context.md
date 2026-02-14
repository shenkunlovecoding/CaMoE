# CaMoE Project Context

## Project Overview

**CaMoE (Capital-driven Mixture of Experts)** is a novel language model architecture based on market economy mechanisms. The main version is **v18 · MiniPile-0.4B · 6R2T-Top2**.

Unlike traditional MoE that relies on static gating or auxiliary losses, CaMoE introduces:
- **Vickrey Auction** (Second-price auction mechanism)
- **Capital Dynamics** & **Progressive Taxation** (Prevents winner-takes-all)
- **Short Selling Mechanism**

This allows RWKV (linear state machine) and Transformer (attention mechanism) experts to achieve natural computational division through free market competition.

---

## Architecture

```
CaMoE_Project/
├── CaMoE/                  # Core architecture modules
│   ├── backbone.py         # RWKV-7 TimeMix backbone + CUDA kernels
│   ├── bridge.py           # UltimateBridge: Low-rank Linear-State Bridge
│   ├── experts.py          # Sparse RWKV & Linear Transformer experts
│   ├── market.py           # CapitalManager + SparseRouter (Vickrey auction)
│   ├── critic.py           # CriticVC: Difficulty prediction + expert affinity
│   ├── system.py           # CaMoE_System & CaMoE_Block main architecture
│   ├── config.py           # v18 configuration (0.4b & 0.1b)
│   ├── config_pilot.py     # Small-scale pilot configs
│   └── wrapper.py          # lm-evaluation-harness adapter
├── scripts/                # Utility scripts
│   ├── prepare_data.py     # Data preprocessing with RWKV tokenizer
│   ├── train_tokenizer.py
│   ├── analyze.py
│   ├── vram_profiler.py
│   └── reset_ckpt.py
├── tokenizer/              # RWKV tokenizer files
├── train.py                # Main training script
├── eval.py                 # Single-sample evaluation with visualization
├── lmeval.py               # lm-evaluation-harness integration
└── README.md
```

---

## Key Concepts

### 1. Market Mechanism
- **Vickrey Auction**: Second-price sealed-bid auction that encourages truthful bidding
- **Capital Dynamics**: Survival of the fittest with progressive taxation to prevent monopoly
- **Top-2 Routing**: Each token is processed by 2 experts with softmax-weighted combination

### 2. Linear-State Bridge
- **Low-Rank Projection**: Projects RWKV's RNN hidden state to Transformer experts
- **O(1) Complexity**: Transformer experts use Linear Cross-Attention (no KV cache backtracking)
- **Gradient Flow**: Bridge allows gradients to flow back to RWKV backbone

### 3. Critic as VC (Venture Capital)
- Predicts token difficulty and expert affinity
- Supports **Long** and **Short** positions
- Penalizes overconfident experts that perform poorly

---

## Expert Configuration

### v18 0.4B (Main)
- **6 RWKV Experts + 2 Transformer Experts (6R2T)**
- Top-2 routing
- 16 layers, 1024 embedding dim
- 65,536 vocab size

### v18 0.1B (Pilot)
- **3 RWKV Experts + 1 Transformer Expert (3R1T)**
- 12 layers, 768 embedding dim
- 32,000 vocab size

---

## Training Pipeline

### 1. Data Preparation
```bash
python scripts/prepare_data.py
```

### 2. Training
```bash
# 0.4B main config
python train.py --scale 0.4b

# 0.1B pilot config
python train.py --scale 0.1b

# Resume from checkpoint
python train.py --scale 0.4b --resume checkpoints/v18_0.4b/step_10000.pth
```

### 3. Training Phases
1. **Prewarm** (first 4K steps): Freeze RWKV, train only Bridge/Transformer/Critic
2. **Warmup** (4K-10K steps): Full parameter warmup
3. **Normal** (after 10K steps): Full training with market updates

### 4. Evaluation
```bash
# Visual single-sample evaluation
python eval.py

# Benchmark with lm-evaluation-harness
python lmeval.py --pretrained checkpoints/v18_0.4b/v18_step2000.pth --tasks arc_easy,hellaswag
```

---

## Core Components

### CaMoE_System (system.py)
Main model class that:
- Manages embedding, blocks, output head
- Coordinates CapitalManager and SparseRouter
- Implements `compute_losses()` and `update_market()`
- Logs market health metrics

### CaMoE_Block (system.py)
Single layer that:
- Runs RWKV-7 TimeMix attention
- Routes tokens to Top-2 experts via market mechanism
- Executes experts with bridge-generated prefix for Transformers

### CapitalManager (market.py)
- Tracks capital for each expert per layer
- Updates capital based on performance vs baseline
- Applies progressive taxation and minimum guarantees

### SparseRouter (market.py)
- Top-2 Vickrey auction routing
- Third-price clearing mechanism
- Adds exploration noise during training

### CriticVC (critic.py)
- Predicts token difficulty (Softplus output)
- Predicts expert affinity (Long/Short signals)
- Settles profits based on prediction accuracy

### UltimateBridge (bridge.py)
- Concatenates current token embedding + RWKV hidden state
- Low-rank compression: [N, 2C] → [N, prefix_len × low_rank_dim]
- Upsampling: [N, prefix_len, low_rank_dim] → [N, prefix_len, C]

---

## Key Design Decisions

1. **Always use market routing** (`use_market=True`) - Even in prewarm/warmup phases
2. **Gradient checkpointing** enabled for training memory efficiency
3. **Mixed precision** (BF16) with CUDA kernels for speed
4. **Capital bailouts** for Critic when capital drops too low
5. **Tied embeddings** for parameter efficiency

---

## Common Tasks

### Adding a New Dataset
1. Add dataset path to `config.py` under `data_roots`
2. Set mixing probability in `mix` dict
3. Run `prepare_data.py` to tokenize

### Modifying Expert Count
Edit in `config.py`:
```python
"num_rwkv_experts": 6,
"num_trans_experts": 2,
"top_k": 2,
```

### Adjusting Market Parameters
```python
"total_capital": 10000.0,
"min_capital_share": 0.05,  # Minimum capital protection
"tax_threshold": 2.0,       # Progressive tax trigger
"tax_rate": 0.1,            # Tax rate
```

---

## Dependencies

Key dependencies (see `requirements.txt`):
- PyTorch 2.0+ (CUDA recommended)
- transformers
- datasets
- bitsandbytes (for 8-bit AdamW)
- swanlab (optional, for experiment tracking)

---

## Benchmarks

TinyStories 0.1B historical results:
- **PPL**: 2.16
- **Speed**: 24,918 TPS (5-8× faster than dense Transformer)
- **Avg Transformer Usage**: 12.59% (depth-wise sparsity)

Emergent layer specialization observed:
- L00-L04: RWKV dominant (lexical/semantic)
- L05: Transformer dominant (logical reasoning pivot)
- L06-L09: RWKV dominant (context maintenance)
- L10-L11: Transformer dominant (output refinement)
