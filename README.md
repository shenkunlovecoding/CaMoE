# CaMoE: Capital-driven Mixture of Experts
*“Not a model, but a living cognitive economy.”*

[中文版](README.zh-CN.md) | [English](README.md)

## 🎯 What is CaMoE?
CaMoE (Capital-driven Mixture of Experts) is a radical rethinking of sparse expert models. Instead of learned routers with auxiliary losses, CaMoE uses zero-parameter Vickrey auctions where experts bid their accumulated capital for the right to process tokens.

**Core insight:** Let market dynamics handle what gradient descent struggles with—load balancing, expert specialization, and adaptive computation.

```
Traditional MoE:  Router(learned) → softmax → top-k → experts
CaMoE:           Auction(zero-param) → capital-based bidding → winner-takes-all
```

## 🚀 Current Version (v22)
### Architecture
```
Input
  │
  ▼
┌─────────────────────────────────────────────┐
│  Embedding + Optional DeepEmbed             │
└─────────────────────────────────────────────┘
  │
  ▼ (× n_layers)
┌─────────────────────────────────────────────┐
│  RWKV-7 TimeMix (Attention Alternative)     │
│  - Linear complexity O(T)                   │
│  - Dynamic state evolution                  │
│  - Custom CUDA kernel (BF16/FP32)           │
├─────────────────────────────────────────────┤
│  CaMoE Block                                │
│  ┌─────────────────────────────────────┐    │
│  │  Vickrey Auction (zero parameters)  │    │
│  │  - bid = capital + α × critic_pos   │    │
│  │  - winner pays second-highest price │    │
│  └─────────────────────────────────────┘    │
│           │                                 │
│           ▼                                 │
│  ┌─────────────────────────────────────┐    │
│  │  Experts (RWKVExpert × n_experts)   │    │
│  │  - LayerNorm → Linear → SiLU → Linear│   │
│  │  - Each owns capital buffer         │    │
│  └─────────────────────────────────────┘    │
│           │                                 │
│  ┌─────────────────────────────────────┐    │
│  │  Critic Pair (variance reduction)   │    │
│  │  - Two critics as mutual baselines  │    │
│  │  - REINFORCE-style training         │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  LM Head (tied weights)                     │
└─────────────────────────────────────────────┘
  │
  ▼
Output
```

### Training Phases
```
Step 0          s1              s2              s3
  │──────────────│───────────────│───────────────│────────▶
     prewarm       market_warm     critic_warm     full_market
     
  [uniform avg]   [auction on]    [critic ramps]  [full system]
   all experts    experts bid       α: 0→1        everything on
```

### Key Components
| File | Description |
| :--- | :--- |
| `model.py` | Main CaMoE_Model with RWKV backbone + expert blocks |
| `block.py` | CaMoE_Block: auction → dispatch → experts |
| `auction.py` | Zero-parameter Vickrey second-price auction |
| `capital.py` | ExpertCapitalManager: settlement, depreciation, EMA |
| `expert_rwkv.py` | RWKVExpert: the actual compute unit |
| `expert_critic.py` | CriticPair: position prediction + REINFORCE |
| `backbone.py` | RWKV-7 TimeMix with custom CUDA kernels |

### Market Mechanics
```python
# Bidding
bid = expert_capital + critic_alpha * critic_position + noise

# Settlement (per token)
profit = (baseline_loss - actual_loss) × capital - price_paid - depreciation
capital_new = capital + profit

# Self-balancing:
# - Rich experts bid high → win more → pay more → high risk
# - Poor experts bid low → find niche → low cost → can recover
```

## 📊 Quick Start
### Installation
```bash
git clone https://github.com/your-repo/camoe.git
cd camoe
pip install torch datasets
```

### Training
```bash
python train.py \
    --scale 0.1b \
    --data /path/to/tokenized_dataset \
    --save_dir checkpoints/ \
    --batch_size 4 \
    --seq_len 512 \
    --steps 10000
```

### Key Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--scale` | 0.4b | Model size: 0.1b or 0.4b |
| `--lr` | config | Expert learning rate |
| `--critic_lr` | config | Critic learning rate |
| `--amp` | off | Enable BF16 mixed precision |
| `--no_compile` | off | Disable torch.compile |

## 📈 Metrics to Watch
```python
# Healthy training signs:
routing_entropy > 1.0        # Multiple experts being used
capital_gini < 0.7           # No monopoly
capital_min > floor          # No mass bankruptcy  
loss decreasing              # Obviously
```
