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

## 🚀 当前版本（v22）
### 架构
```
输入
  │
  ▼
┌─────────────────────────────────────────────┐
│  Embedding + 可选 DeepEmbed                 │
└─────────────────────────────────────────────┘
  │
  ▼（× n_layers）
┌─────────────────────────────────────────────┐
│  RWKV-7 TimeMix（Attention 替代）           │
│  - 线性复杂度 O(T)                          │
│  - 动态状态演化                             │
│  - 自定义 CUDA kernel（BF16/FP32）          │
├─────────────────────────────────────────────┤
│  CaMoE Block                                │
│  ┌─────────────────────────────────────┐    │
│  │  Vickrey 拍卖（零参数）             │    │
│  │  - bid = capital + α × critic_pos   │    │
│  │  - winner 支付第二高价格            │    │
│  └─────────────────────────────────────┘    │
│           │                                 │
│           ▼                                 │
│  ┌─────────────────────────────────────┐    │
│  │  Experts（RWKVExpert × n_experts）  │    │
│  │  - LayerNorm → Linear → SiLU → Linear│   │
│  │  - 每个专家都有自己的资本 buffer    │    │
│  └─────────────────────────────────────┘    │
│           │                                 │
│  ┌─────────────────────────────────────┐    │
│  │  Critic Pair（方差降低）            │    │
│  │  - 两个 critic 互为 baseline        │    │
│  │  - 使用 REINFORCE 风格训练          │    │
│  └─────────────────────────────────────┘    │
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

### 训练阶段
```
Step 0          s1              s2              s3
  │──────────────│───────────────│───────────────│────────▶
     prewarm       market_warm     critic_warm     full_market

  [uniform avg]   [auction on]    [critic ramps]  [full system]
   所有专家训练     专家开始竞标      α: 0→1         全系统开启
```

### 关键组件
| 文件 | 说明 |
| :--- | :--- |
| `model.py` | 主体 `CaMoE_Model`，包含 RWKV backbone 与 expert blocks |
| `block.py` | `CaMoE_Block`：auction → dispatch → experts |
| `auction.py` | 零参数 Vickrey 二价拍卖 |
| `capital.py` | `ExpertCapitalManager`：结算、折旧、EMA |
| `expert_rwkv.py` | `RWKVExpert`：实际执行计算的专家 |
| `expert_critic.py` | `CriticPair`：仓位预测 + REINFORCE |
| `backbone.py` | 带自定义 CUDA kernel 的 RWKV-7 TimeMix |

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
