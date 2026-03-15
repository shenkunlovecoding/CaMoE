# CaMoE

CaMoE（Capital-driven Mixture of Experts）是一个把稀疏专家路由改写成预测市场的实验仓库。当前主线版本是 `v23.0`：每一层同时包含序列市场和 FFN 市场，专家不再只靠 router logits 决定输赢，而是基于钱包、价格和 `RewardCritic` 预测的回报进行 Top-1 竞争。

这个仓库同时内置了三套 ROSA 家族实现：

- `wind_rosa/`：硬符号匹配 CUDA 路径
- `rosa_soft/`：proxy / SUFA / scan 路径
- `Soft_ROSA/`：exact soft DP 与 QKV-1bit 实验实现

如果你想先理解当前设计，再看更长的研究笔记，建议顺序是：

- `CHANGELOG.md`
- `NOTE.zh-CN.md`
- `NOTE.md`

## 当前实现概览

- 双市场 block
  - 序列市场：`TimeMixExpert` vs `ROSAExpert`
  - FFN 市场：`RWKVExpert` / `DeepEmbedExpert` / `SlimDeepEmbedExpert`
- 统一的 `PredictionMarketRouter`
- 统一的 `MarketStateManager`
- 每个市场一个监督式 `RewardCritic`
- 训练期硬 Top-1 + STE 软混合，推理期严格硬 Top-1
- 两阶段训练调度：`uniform_warmup -> full_market`
- vendored ROSA 后端、benchmark、toy 数据脚本和单元测试

## 核心路由机制

在 `v23.0` 中，每层、每个市场都会显式维护：

```python
wallet[i]   # 代码里沿用 capital 这个 buffer 名字，本质是专家预算
q[i]        # 市场信念状态
price[i] = (1 - liquidity_floor) * softmax(q / T) + liquidity_floor / K
```

每个 batch 开始时，先根据当前钱包和价格给专家定价：

```python
stake[i]  = bet_fraction * wallet[i]
shares[i] = stake[i] / price[i]
```

随后对每个 token 计算预期收益并选出 winner：

```python
pred_reward[t, i] = sigmoid(reward_critic(x_t)[i])
score[t, i] = shares[i] * (pred_reward[t, i] - price[i])
winner[t] = argmax(score[t, i])
```

训练期还可以叠加两种探索：

- `routing_noise_std`：给 score 加高斯噪声
- `exploration_epsilon`：按概率随机覆盖 winner

结算时使用 token 级利润：

```python
reward[t] = sigmoid(
    reward_scale * (market_loss_ema - token_loss[t]) / (abs(market_loss_ema) + reward_eps)
)

token_profit[t] = shares[winner[t]] * (reward[t] - price[winner[t]])
```

几个实现细节需要特别注意：

- `shares` 在 batch 开始时定价一次，整批 token 复用
- `loss_ema` 是 market 级共享量，不是 per-expert
- 没有赢到 token 的专家不会更新钱包和 `q`

## 训练流程

默认流程已经从旧版四阶段收敛成两阶段：

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

## ROSA 后端总览

`camoe/config.py` 里已经把当前支持的后端名收敛成 canonical backend，同时保留了旧 alias。CLI 里既可以写 canonical 名，也可以继续写旧别名。

| 类别 | 常用别名 | 代表 backend | 说明 |
| :--- | :--- | :--- | :--- |
| 硬符号多比特 | `wind` | `hard_symbolic_multibit` | 当前默认路径，走 vendored `wind_rosa/` |
| proxy soft match | `soft` | `soft_match` | 走 `rosa_soft` 的 soft match 近似 |
| proxy suffix | `sufa` | `soft_suffix` | 走 `rosa_soft` 的 SUFA 近似 |
| proxy suffix scan | `scan` | `soft_suffix_scan` | 走 `rosa_soft` 的 scan 路径 |
| exact soft DP | `soft_exact` | `soft_exact_dp` | 走 `Soft_ROSA/` exact DP，自动选 scan backend |
| exact soft DP 指定后端 | `soft_exact_serial` / `soft_exact_cuda` / `soft_exact_triton` | `soft_exact_dp_*` | 强制串行、CUDA 或 Triton diagonal scan |
| 多比特 exact QKV | `soft_qkv_multibit_exp*` 等旧名 | `soft_qkv_multibit*` / `soft_qkv_multibit_unmatched*` / `hard_qkv_multibit` | 先对 q/k/v 做符号化，再走 exact 或 hard 路径 |
| 1-bit 专项 | `soft_qkv1bit*` 等旧名 | `soft_qkv_binary*` / `soft_qkv_binary_bipolar*` | 面向 `rosa_bits == 1` 的实验路径 |

如果你只是想先把系统跑起来，建议优先试这几种：

- `hard_symbolic_multibit`
- `soft_exact_dp`
- `soft_exact_dp_cuda`
- `soft_qkv_binary_cuda`（仅 `rosa_bits == 1`）

## 环境与依赖

这个仓库依赖 PyTorch、CUDA 编译工具链以及若干运行时扩展。第一次调用某些后端时，代码会动态编译 CUDA / C++ / Triton 扩展，所以环境是否完整非常重要。

推荐注意事项：

- Windows 工作流可以先运行根目录的 `init.bat`
  - 会加载 VS2022 x64 Build Tools
  - 会激活名为 `CaMoE` 的 conda 环境
- `requirements.txt` 更接近“当前开发环境快照”，不是严格的最小依赖清单
- 其中包含一个本地 DeepSpeed wheel 路径：

```text
deepspeed @ file:///G:/DeepSpeed/dist/...
```

如果你不是在同一台机器上复现，需要手动替换、删除或自己安装 DeepSpeed。

还需要知道：

- `camoe/backbone.py` 会在首次运行时编译 RWKV-7 CUDA kernel
- `Soft_ROSA/soft_rosa_cuda.py` 和 `Soft_ROSA/qkv1bit_cuda.py` 会在首次使用 CUDA backend 时编译扩展
- Triton 是可选项；没有 Triton 时，相关路径会跳过或 fallback
- `swanlab` 也是可选项；不想启用时在 toy 训练里加 `--no-swanlab`

## 快速开始

### 1. 初始化环境

Windows:

```powershell
.\init.bat
```

依赖安装：

```powershell
pip install -r requirements.txt
```

如果 `requirements.txt` 里的本地 wheel 路径不可用，请先修正那一行再安装。

### 2. 生成 toy 数据集

`scripts/make_reverse_digits_data.py` 会生成一个 `DatasetDict`，默认保存到 `data/<task>`。可选任务包括 `reverse_digits`、`pattern_complete`、`mixed_v3`、`mixed_all_digits` 等。

```powershell
python scripts\make_reverse_digits_data.py `
  --task mixed_v3 `
  --output data\mixed_v3 `
  --print-task-intro
```

### 3. 跑 toy 训练与路由可视化

```powershell
python scripts\train_reverse_digits.py `
  --data_dir data\mixed_v3 `
  --task_filter mixed_v3 `
  --model_kind camoe `
  --steps 10000 `
  --rosa_backend soft_exact_dp `
  --save_dir checkpoints\mixed_v3 `
  --artifact_dir artifacts\mixed_v3 `
  --no-swanlab
```

如果你想只看单个子任务，也可以把 `--task_filter` 改成例如 `pattern_complete` 或 `delayed_copy`。

### 4. 准备通用语言数据

`train.py` 期望输入是一个由 `datasets.load_from_disk()` 读取的 Hugging Face 数据集目录，且训练 split 至少包含 `input_ids` 字段。

仓库自带的 `scripts/prepare_data.py` 是一个非流式数据预处理脚本，但它的配方是写死在文件顶部 `DATA_RECIPE` 里的。实际使用前请先按你的数据源修改它，再运行：

```powershell
python scripts\prepare_data.py `
  --save_path data\camoe_mix_v20 `
  --ctx_len 1024 `
  --num_proc 4 `
  --batch_size 100
```

### 5. 主训练

```powershell
python train.py `
  --scale 0.1b `
  --data data\camoe_mix_v20 `
  --save_dir checkpoints\v23 `
  --batch_size 4 `
  --seq_len 512 `
  --steps 10000 `
  --bet_fraction 0.05 `
  --price_lr 0.02 `
  --liquidity_floor 0.02 `
  --reward_scale 5.0 `
  --exploration_epsilon 0.02 `
  --rosa_backend hard_symbolic_multibit
```

常用可调参数：

- `--critic_lr`
- `--critic_update_interval`
- `--uniform_warmup_steps`
- `--market_ramp_steps`
- `--routing_noise_std`
- `--routing-ste/--no-routing-ste`
- `--rosa_hard_backend`
- `--rosa_hard_switch_step`
- `--rosa_bits`
- `--slim_rosa_heads`
- `--rosa_truncation_length`

### 6. 生成与评测

单条 prompt 生成：

```powershell
python eval.py `
  --checkpoint checkpoints\v23\v23_final.pth `
  --prompt "Hello" `
  --device cuda
```

`eval.py` 会同时打印生成文本和每层最后一步的 winner 统计。

LM Evaluation Harness：

```powershell
python lmeval.py `
  --pretrained checkpoints\v23\v23_final.pth `
  --tasks lambada_openai `
  --batch_size 1 `
  --device cuda
```

## Benchmark 与测试

ROSA 后端端到端 benchmark：

```powershell
python scripts\benchmark_rosa_backends.py --device cuda
```

prediction-market 单元测试：

```powershell
python -m unittest tests.test_prediction_market -v
```

SUFA kernel 测试：

```powershell
python -m unittest tests.test_rosa_sufa_kernels -v
```

语法级 smoke check：

```powershell
python -m compileall camoe train.py scripts tests Soft_ROSA
```

## 仓库结构

| 路径 | 说明 |
| :--- | :--- |
| `camoe/` | 主模型、市场机制、专家实现与 adapter |
| `train.py` | 主训练入口 |
| `eval.py` | 文本生成与路由统计 |
| `lmeval.py` | `lm-evaluation-harness` 入口 |
| `scripts/make_reverse_digits_data.py` | toy 数据生成 |
| `scripts/train_reverse_digits.py` | toy 训练、评估、可视化 |
| `scripts/benchmark_rosa_backends.py` | 比较 ROSAExpert 后端速度 |
| `tests/test_prediction_market.py` | prediction-market 核心测试 |
| `tests/test_rosa_sufa_kernels.py` | SUFA kernel 测试 |
| `wind_rosa/` | vendored 硬符号 ROSA |
| `rosa_soft/` | vendored proxy / SUFA / scan 实现 |
| `Soft_ROSA/` | vendored exact Soft ROSA 与 QKV-1bit 实现 |

## 备注

- 代码中的 `capital` 在 `v23` 里应理解成预算或钱包，不再是旧版意义上的直接收益倍率。
- 推理阶段使用严格硬 Top-1 路由。
- 旧版 Vickrey / REINFORCE checkpoint 不保证能直接加载。
- 这个仓库目前更偏研究与实验，不是做成可直接 `pip install` 的通用库。
