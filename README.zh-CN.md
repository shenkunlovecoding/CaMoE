# CaMoE：资本驱动的专家混合
*用预测市场来做稀疏专家路由。*

[English](README.md) | [中文版](README.zh-CN.md)

## 什么是 CaMoE？
CaMoE 是一种把稀疏 MoE 路由看作预测市场的问题设定，而不是再训练一个带辅助平衡损失的 Router。

当前实现里，每个市场都维护：
- 每个专家的钱包 / 预算（沿用 `capital` 这个 buffer 名字）
- 市场信念状态 `q`
- 满足 `sum(p)=1` 的价格 `p`
- 一个监督式 `RewardCritic`，预测 token 在每个专家上的结算回报

当前路由路径：

```text
传统 MoE： learned router -> logits -> top-k experts
CaMoE v23： 市场价格 + 专家钱包 + reward critic -> top-1 winner
```

## 当前版本
`v23.0` 是 prediction-market 重写版。当前运行路径已经不再使用：
- Vickrey 二价拍卖
- critic pair
- REINFORCE 结算
- `prewarm / market_warm / critic_warm / full_market` 四阶段流程

现在使用的是：
- 序列市场和 FFN 市场共享同一种 prediction-market 路由语义
- 监督式 `RewardCritic`
- token 级 `shares * (reward - price)` 结算
- 两阶段训练：`uniform_warmup` -> `full_market`

## 架构

```text
输入
  |
  v
Embedding
  |
  v  x n_layers
+------------------------------------------------------+
| 序列市场                                             |
| - TimeMixExpert vs ROSAExpert                        |
| - 共享 prediction-market 路由逻辑                    |
| - 前向硬 Top-1，训练期 STE 软混合                    |
+------------------------------------------------------+
| FFN 市场                                             |
| - RWKVExpert / DeepEmbed / SlimDeepEmbed             |
| - 同一套 prediction-market 路由                      |
| - 前向硬 Top-1，训练期 STE 软混合                    |
+------------------------------------------------------+
| Reward Critics                                       |
| - 每个市场一个监督式 critic                          |
| - 预测 token 在各专家上的回报                        |
+------------------------------------------------------+
  |
  v
LM Head
  |
  v
输出
```

## 市场机制
每层、每个市场维护：

```python
wallet[i]   # 专家预算，运行时仍存放在 capital buffer 里
q[i]        # 市场信念状态
price[i] = (1 - liquidity_floor) * softmax(q / T) + liquidity_floor / K
```

路由：

```python
stake[i]  = bet_fraction * wallet[i]
shares[i] = stake[i] / price[i]          # 每个 batch 开始时计算一次

pred_reward[t, i] = sigmoid(reward_critic(x_t)[i])
score[t, i] = shares[i] * (pred_reward[t, i] - price[i])

winner[t] = argmax(score[t])
```

训练期探索：
- `routing_noise_std`：给路由分数加高斯噪声
- `exploration_epsilon`：按概率随机覆盖 winner

结算：

```python
reward[t] = sigmoid(
    reward_scale * (market_loss_ema - token_loss[t]) / (abs(market_loss_ema) + reward_eps)
)

token_profit[t] = shares[winner[t]] * (reward[t] - price[winner[t]])
wallet_update[i] = 对 batch 内该专家赢下的 token_profit 做加权均值
q[i] += price_lr * (avg_reward_i - price[i])   # 只更新赢到 token 的专家
```

当前实现里几个关键约束：
- `shares` 在 batch 开始时按当下钱包和价格定价一次，然后整批 token 复用。
- `loss_ema` 是按 market 共享，而不是 per-expert。
- 某个专家如果这一批没有赢到 token，它的钱包和 `q` 都不变。

## 训练流程
默认训练流程是两阶段：

```text
Step 0                     uniform_warmup_steps                结束
  |-----------------------------------|------------------------->
          uniform_warmup                          full_market
```

`uniform_warmup`
- 输出走均匀混合
- 不更新钱包
- 不更新价格
- 不训练 reward critic

`full_market`
- 开启 prediction-market 路由
- 开启钱包结算
- 开启价格更新
- 每步训练 reward critic

## 仓库结构
| 文件 | 说明 |
| :--- | :--- |
| `camoe/model.py` | 主模型 `CaMoE_Model` 与市场结算主循环 |
| `camoe/block.py` | 单层里的序列市场和 FFN 市场路由 |
| `camoe/auction.py` | `PredictionMarketRouter` |
| `camoe/capital.py` | `MarketStateManager`：钱包、价格、共享 EMA |
| `camoe/expert_critic.py` | `RewardCritic` |
| `camoe/expert_timemix.py` | TimeMix 序列专家 |
| `camoe/expert_rosa.py` | Wind ROSA 序列专家 |
| `camoe/expert_rwkv.py` | RWKV FFN 专家 |
| `camoe/expert_deepembed.py` | DeepEmbed 专家 |
| `train.py` | 主训练入口 |
| `scripts/train_reverse_digits.py` | toy 任务训练与路由可视化 |
| `tests/test_prediction_market.py` | prediction-market 核心单测 |

## 快速开始
### 安装
```bash
pip install -r requirements.txt
```

### 训练
```bash
python train.py \
    --scale 0.1b \
    --data /path/to/tokenized_dataset \
    --save_dir checkpoints/v23 \
    --batch_size 4 \
    --seq_len 512 \
    --steps 10000 \
    --bet_fraction 0.05 \
    --price_lr 0.02 \
    --liquidity_floor 0.02 \
    --reward_scale 5.0 \
    --exploration_epsilon 0.02
```

### Toy 训练 / 路由可视化
```bash
python scripts/train_reverse_digits.py \
    --model_kind camoe \
    --steps 10000 \
    --no_swanlab
```

### 生成
```bash
python eval.py \
    --checkpoint /path/to/checkpoint.pth \
    --prompt "Hello" \
    --device cuda
```

### LM Eval Harness
```bash
python lmeval.py \
    --pretrained /path/to/checkpoint.pth \
    --tasks lambada_openai \
    --device cuda
```

## 重要参数
| 参数 | 含义 |
| :--- | :--- |
| `--bet_fraction` | 每次前向把钱包的多少比例转成 stake |
| `--price_lr` | 市场信念 `q` 对真实回报的响应速度 |
| `--price_temperature` | 市场价格 softmax 温度 |
| `--liquidity_floor` | 防止专家价格塌到 0 |
| `--reward_scale` | 从 loss 改善映射到 reward 的陡峭程度 |
| `--reward_eps` | reward 计算的数值稳定项 |
| `--reward_hidden_dim` | reward critic 隐层宽度 |
| `--routing_noise_std` | 训练期加在 score 上的高斯噪声 |
| `--exploration_epsilon` | 训练期随机覆盖 winner 的概率 |
| `--uniform_warmup_steps` | 均匀热身阶段长度 |
| `--routing_ste` | 是否启用 STE 软混合 |

## 建议观察的指标
健康训练通常表现为：

```text
price_max 没有直接冲到绝对垄断
wallet_gini 保持有界
wallet_min 始终高于 floor
好的专家 realized_reward_mean 能长期高于自身价格
expected_profit_mean 不是长期贴近 0
routing_entropy 不会过早塌缩
exploration_rate 接近配置值
主 LM loss 持续下降
```

当前日志已经从旧版的 bid/capital/critic-alpha 视角，切到 wallet/price/reward 视角。

## 测试
核心 prediction-market 单测：

```bash
python -m unittest tests.test_prediction_market -v
```

静态语法检查：

```bash
python -m compileall camoe train.py scripts/train_reverse_digits.py tests
```

## 说明
- 运行时 buffer 仍叫 `capital`，但在 `v23` 里它只表示预算，不再直接充当收益乘数。
- Vickrey / REINFORCE 时代的旧 checkpoint 不保证可以平滑加载。
- 某些辅助脚本里可能还保留旧 CLI 参数做兼容，但它们不属于 `v23` 的核心路由语义。
- 推理阶段使用严格硬 Top-1 路由。
