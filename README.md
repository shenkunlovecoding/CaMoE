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

## 🚀 Current Version (v22.1)
### Architecture
```
Input
  │
  ▼
┌─────────────────────────────────────────────┐
│  Embedding                                  │
└─────────────────────────────────────────────┘
  │
  ▼ (× n_layers)
┌─────────────────────────────────────────────┐
│  Sequence Market                            │
│  - TimeMixExpert vs ROSAExpert              │
│  - Full-sequence state update for both      │
│  - Winner output with STE backward          │
├─────────────────────────────────────────────┤
│  FFN Market                                 │
│  - RWKVExpert / DeepEmbed / SlimDeepEmbed   │
│  - Vickrey winner-takes-all                 │
│  - STE during training, hard winner at eval │
├─────────────────────────────────────────────┤
│  Critic Pair                                │
│  - REINFORCE update                         │
│  - Entropy regularization (optional)        │
│  - Shadow training in prewarm/market_warm   │
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

### Training Phases (default toy recipe)
```
Step 0        1500         3000             5500+
  │────────────│────────────│────────────────│────────▶
    prewarm      market_warm   critic_warm      full_market

  uniform=True   auction on     critic α ramps   full system
  (no hard route)  + STE        + STE anneal      hard eval route

STE temperature anneal:
  2.0  -> 1.0 -> 0.3
```

### Key Components
| File | Description |
| :--- | :--- |
| `model.py` | Main `CaMoE_Model` with dual-market blocks |
| `block.py` | Sequence market + FFN market routing logic |
| `auction.py` | Zero-parameter Vickrey second-price auction |
| `capital.py` | ExpertCapitalManager: settlement, depreciation, EMA |
| `expert_timemix.py` | TimeMix expert for sequence market |
| `expert_rosa.py` | Slim Wind ROSA sequence expert |
| `expert_rwkv.py` | RWKV FFN expert |
| `expert_critic.py` | CriticPair: position prediction + REINFORCE |
| `expert_deepembed.py` | DeepEmbed experts in FFN market |

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
