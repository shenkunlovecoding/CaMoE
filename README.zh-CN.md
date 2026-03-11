# CaMoE：资本驱动的专家混合
*“这不只是一个模型，而是一个活的认知经济体。”*

[English](README.md) | [中文版](README.zh-CN.md)

## 🎯 什么是 CaMoE？
CaMoE（Capital-driven Mixture of Experts，资本驱动的专家混合）是对稀疏专家模型的一次激进重构。它不再使用带辅助损失的可学习 Router，而是使用零参数的 Vickrey 拍卖，让专家用自己积累的资本竞标处理 token 的权利。

**核心洞见：** 把梯度下降不擅长处理的问题交给市场动力学，例如负载均衡、专家专精和自适应计算。

```
传统 MoE:  Router(learned) → softmax → top-k → experts
CaMoE:    Auction(zero-param) → 基于资本竞价 → winner-takes-all
```

## 🚀 当前版本（v22.1）
### 架构
```
输入
  │
  ▼
┌─────────────────────────────────────────────┐
│  Embedding                                  │
└─────────────────────────────────────────────┘
  │
  ▼（× n_layers）
┌─────────────────────────────────────────────┐
│  序列市场                                   │
│  - TimeMixExpert vs ROSAExpert              │
│  - 两者都吃完整序列更新状态                 │
│  - 前向硬路由，反向 STE                     │
├─────────────────────────────────────────────┤
│  FFN 市场                                   │
│  - RWKVExpert / DeepEmbed / SlimDeepEmbed   │
│  - Vickrey winner-takes-all                 │
│  - 训练期 STE，评估期纯硬路由               │
├─────────────────────────────────────────────┤
│  Critic Pair                                │
│  - REINFORCE 更新                           │
│  - 可选路由熵正则                           │
│  - prewarm/market_warm 影子训练             │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  LM Head（权重共享）                        │
└─────────────────────────────────────────────┘
  │
  ▼
输出
```

### 训练阶段（toy 默认配置）
```
Step 0        1500         3000             5500+
  │────────────│────────────│────────────────│────────▶
    prewarm      market_warm   critic_warm      full_market

  uniform=True   auction on     critic α 渐进     全系统开启
  （无硬路由）    + STE          + STE 退火

STE 温度退火：
  2.0  -> 1.0 -> 0.3
```

### 关键组件
| 文件 | 说明 |
| :--- | :--- |
| `model.py` | 双市场主模型 `CaMoE_Model` |
| `block.py` | 序列市场 + FFN 市场路由核心 |
| `auction.py` | 零参数 Vickrey 二价拍卖 |
| `capital.py` | `ExpertCapitalManager`：结算、折旧、EMA |
| `expert_timemix.py` | 序列市场 TimeMix 专家 |
| `expert_rosa.py` | Slim Wind ROSA 专家 |
| `expert_rwkv.py` | RWKV FFN 专家 |
| `expert_critic.py` | `CriticPair`：仓位预测 + REINFORCE |
| `expert_deepembed.py` | FFN 市场 DeepEmbed 专家 |

### 市场机制
```python
# 竞价
bid = expert_capital + critic_alpha * critic_position + noise

# 结算（按 token）
profit = (baseline_loss - actual_loss) × capital - price_paid - depreciation
capital_new = capital + profit

# 自平衡：
# - 富专家出价高 → 赢得更多 → 支付更多 → 风险更高
# - 穷专家出价低 → 寻找细分生态位 → 成本更低 → 仍可恢复
```

## 📊 快速开始
### 安装
```bash
git clone https://github.com/your-repo/camoe.git
cd camoe
pip install torch datasets
```

### 训练
```bash
python train.py \
    --scale 0.1b \
    --data /path/to/tokenized_dataset \
    --save_dir checkpoints/ \
    --batch_size 4 \
    --seq_len 512 \
    --steps 10000
```

### 关键参数
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--scale` | 0.4b | 模型规模：0.1b 或 0.4b |
| `--lr` | config | Expert 学习率 |
| `--critic_lr` | config | Critic 学习率 |
| `--amp` | off | 启用 BF16 混合精度 |
| `--no_compile` | off | 禁用 `torch.compile` |

## 📈 建议观察的指标
```python
# 健康训练的信号：
routing_entropy > 1.0        # 多个专家都在被使用
capital_gini < 0.7           # 没有形成垄断
capital_min > floor          # 没有大面积破产
loss decreasing              # 显然应该下降
```
