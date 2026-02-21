# CaMoE: Capital-driven Mixture of Experts

> "We choose to go to the moon... not because they are easy, but because they are hard."

https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg](https://opensource.org/licenses/MPL-2.0)
https://img.shields.io/badge/Architecture-Hybrid_MoE-blueviolet](https://github.com/shenkunlovecoding/CaMoE)
https://img.shields.io/badge/Speed-7k_TPS-orange](https://github.com/shenkunlovecoding/CaMoE)

**CaMoE (Capital-driven Mixture of Experts)** 是一个基于**市场经济机制**的混合专家语言模型架构，目前主力版本为 **v21.1 · 0.4B · 6R2T-Top2**。

不同于传统 MoE 依赖静态门控或纯辅助损失（Auxiliary Loss），CaMoE 将路由拆分为两个协作系统：

- **梯度系统**：可微的 Gate 网络，通过反向传播学习 token 级路由偏好；
- **市场系统**：基于 Vickrey 拍卖与资本动态的全局资源分配器，作为 Gate 网络的“顾问”。

两个系统各司其职：梯度管“这个 token 该怎么路由”，市场管“全局资源怎么分配”。RWKV（线性状态机）和 Transformer（注意力机制）专家在两套信号的共同引导下实现算力的自然分工。

---

## ✅ 近期更新

### v21.1（2026-02，当前版本）

- 修复 `grad_accum` 在 resume 场景下的潜在残留梯度问题（训练循环启动即 `zero_grad`）。
- 修复 `market_update` 仅使用最后一个 micro-batch 的偏差，改为按 `grad_accum` 窗口聚合更新。
- 文档澄清：`LinearTransformerExpert` 当前为 SDPA softmax prefix-attention；`SparseRouter` 为 Top-K 输出稀疏（非前置粗筛）。
- 增加模块级 `torch.compile` 加速实验（`Bridge / Critic / Experts`），并同步依赖列表（`requirements.txt`）。

### v21（2026-02）

**1. Gradient Gate × Market Bias 双通道路由**

v20 及之前版本的核心问题：路由决策完全依赖市场资本，`weights.detach()` 切断了梯度——语言模型的 CE loss 无法反向传播到路由选择，导致 Router 只能依赖统计信号（市场资本增减）学习，在实际训练中表现为**路由坍塌**（多数层锁死在 0% 或 100% Transformer 使用率，且与输入内容无关）。

v21 将路由重构为双通道架构：

```
┌──────────────┐    ┌──────────────┐
│ Gradient Gate │    │ Market Bias   │
│ (可微,依赖    │    │ (不可微,全局  │
│ 输入内容)     │    │ 资本信号)     │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
gate_logits + α · capital_bias.detach()
       │                   │
       └────────┬──────────┘
                ▼
        adjusted_logits
                │
        ┌───────┴────────┐
        ▼                ▼
Top-K Selection     Soft Weights
(离散,detach)      (连续,可微!)
winners        F.softmax(gate_logits[top_k])
        │
        ▼
Σ weight_i × expert_i(x)
        │
        ▼ 梯度流回 gate 网络 ✅
       loss
```

- **选举与加权分离**：`winners = topk(adjusted_logits + noise)`（离散、detach，市场参与选举）；`weights = softmax(gate_logits[top_k])`（连续、可微，纯梯度决定权重分配）。
- **Market Influence（非梯度自适应）**：每层一个标量 `alpha_l`（buffer），初始化较小（`market_alpha_init=0.05`），并在 `update_market` 中基于 Gate/Market 一致性用 EMA 规则自适应更新。
- **Load Balancing Auxiliary Loss**：防止路由坍塌的安全网，统计每个 expert 被选中的频率并惩罚偏离均匀分布的程度（系数 `aux_loss_coeff=0.01`）。
- **训练噪声策略**：仅训练态注入高斯噪声（`router_noise_std=0.02`），评估/推理不加噪声。

**2. 七阶段训练调度（SFT/RLHF 占位）**
`prewarm(2k) → warm(3k) → criticwarm(4k) → prenormal(3k) → normal(40k) → sft(0) → rlhf(0)`

| 阶段 | 步数 | 训练范围 | 市场 | 梯度路由 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| prewarm | 2,000 | router_conf + bridge | ❌ | ✅ | 纯梯度预热 Gate 网络 |
| warm | 3,000 | all (lr×0.35) | ✅ | ✅ | 全模型低学率预热 |
| criticwarm | 4,000 | critic only (lr×2.0) | ✅ | ✅ | Critic 高学率训练 |
| prenormal | 3,000 | all (分组学率) | ✅ | ✅ | 过渡到全速 |
| normal | 40,000 | all (lr×1.0) | ✅ | ✅ | 主训练阶段 |
| sft | 占位 | all | ✅ | ✅ | 监督微调（未启用） |
| rlhf | 占位 | all | ✅ | ✅ | RLHF（未启用） |

**3. 经济系统增强（延续 v20 并与 v21 路由解耦）**

市场系统从 v20 的“独裁者”（唯一路由决策者）转变为“顾问”（提供 `capital_bias` 给 Gate 网络）。所有经济子系统保留但角色调整：

| 子系统 | v20 角色 | v21 角色 |
| :--- | :--- | :--- |
| Capital Manager | 直接决定路由 | 提供 selection bias |
| Vickrey Auction | 唯一路由机制 | 辅助选举偏置 |
| QE/QT | 控制路由流量 | 维护全局资本健康 |
| Critic/VC | 影响路由 | 提供难度信号 + 风投注资 |
| Idle Tax / Depreciation | 惩罚不活跃专家 | 同上（但梯度可以独立救活专家） |

**新增子系统：**
- **中央银行与 QE/QT**：按层总资本偏离阈值注入/回收流动性
- **基础算力保障**：每个专家维持最低资本线（`base_compute_floor_ratio`）
- **破产保护与债务重组**：Critic 触发救助后记录债务并自动偿还
- **破产权重漂移**：向高表现 donor 层 Critic 参数做插值迁移
- **风投注资（VC Injection）**：高 affinity + 低资本 + 低选中率的专家获得额外注资
- **监控指标**：`MarketAlpha`、`WinnerFromAdjustedEntropy`、`WeightEntropy`、`AssetVelocity`

### v20（2026-01）
- 七阶段调度原型、CriticWarm 机制、经济系统基础版

### v19（2026-01）
- RWKV-7 ClampW Kernel BF16 兼容性修复
- Loss 口径修正（padding `-100`、tied embedding rescale）
- NaN 诊断链路（逐层逐张量定位）

### v10–v18（2025-12 ~ 2026-01）
- 市场机制原型、Vickrey 拍卖、Linear-State Bridge、MiniPile 0.4B 实验

---

## 🏆 Benchmark

### 0.1B TinyStories Pilot（Greedy + repetition penalty=1.2）

> 设置：纯 TinyStories Pilot，Greedy 解码，`repetition_penalty=1.2`，6 组英文 prompt 验证。

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Samples** | **6 prompts / 533 tokens** | 覆盖常识补全、故事续写、名句开头 |
| **Global Routing** | **Mixed 100%** | Pure RWKV=0%，Deep Trans=0% |
| **Generation Quality** | **可读稳定，但偏 TinyStories 文风** | 叙事连贯，知识问答题会被故事化 |

### 层级路由画像（本轮 6 prompt 平均 Top-2 命中率）
CaMoE 在本次 Pilot 中呈现“关键层强 Trans、中间层混合、个别层近关闭”的结构：

```
L00  100.0% | L01   93.0% | L02   0.0% | L03   0.2%
L04   63.7% | L05    0.1% | L06  49.6% | L07  40.4%
L08   64.3% | L09  100.0% | L10  42.3% | L11  22.6%
```

结论：路由并未坍塌到单一路径，而是形成分层分工；但在纯 TinyStories 语料下，开放域事实提示词会被“儿童故事模板”覆盖。

### v21.1 0.4B 训练状态
> ⚠️ v21.1 目前为 architecture iteration 阶段，尚未完成完整 14B token 训练。
> 当前已验证梯度流通与路由动态性，完整 benchmark 待后续更新。

---

## 🌟 核心创新

### 1. Dual-Path Router（双通道路由）
**设计哲学：梯度负责精确，市场负责全局。**

| | 梯度系统 | 市场系统 |
| :--- | :--- | :--- |
| **学什么** | **这个 token** 该用哪个 expert | **全局来看** 哪些 expert 更有价值 |
| **时间尺度** | 每步更新 | 缓慢漂移（EMA） |
| **信号来源** | CE loss 反向传播 | token_loss 统计 + 资本增减 |
| **作用方式** | 决定 soft weights | 提供 selection bias |
| **可微性** | ✅ | ❌（不需要） |

**两个系统的协作动力学：**

- **Step 0-2000 (早期)**: gate ≈ 0 → 路由接近均匀 + 噪声，所有 expert 都能训练到；market capital 也均匀 → gate 网络开始学习 input-dependent 模式。
- **Step 2000-10000 (中期)**: gate 开始分化：不同输入路由到不同 expert；market 也开始分化：表现好的 expert 积累更多资本；两者是否一致？一致 → 正反馈，路由稳定；不一致 → market_influence (α) 会被由 EMA 规则压低。
- **Step 10000+ (后期)**: gate 已经学会 input-dependent 路由；market 提供全局先验；α 稳定在某个值 → 两个系统达到均衡。

### 2. Market Mechanism（市场机制）
- **Vickrey 拍卖**：Top-3 选举中第 3 名的分数作为 cost，激励诚实报价。
- **资本动态 & 累进税**：防止马太效应，无需纯 Auxiliary Loss 即可辅助负载均衡。
- **Load Balance Auxiliary Loss**：作为安全网，与市场机制互补。

### 3. Linear-State Bridge（线性状态桥）
- **Low-Rank Projection**：将 RWKV 的 RNN Hidden State 通过低秩投影 `[N, 2C] → [N, P, r] → [N, P, C]` 生成前缀。
- **O(1) Complexity**：Transformer 专家使用 Linear Cross-Attention（Query=token, K/V=prefix），不回溯历史 KV Cache。

### 4. Critic as VC（风投模式）
- 预测 token 级难度（`difficulty`）与专家适配度（`affinity`）。
- 支持做多/做空：如果专家过度自信但答错，Critic 通过做空剥夺其资本。
- 破产保护：触发救助后记录债务，从高表现层参数漂移重组。

---

## 📁 项目结构

```
CaMoE_Project/
├── CaMoE/
│   ├── backbone.py      # RWKV-7 TimeMix + DeepEmbedAttention + CUDA Kernel
│   ├── bridge.py        # UltimateBridge：低秩 Linear-State Bridge
│   ├── experts.py       # SparseRWKVFFN + LinearTransformerExpert（含 confidence 网络）
│   ├── market.py        # CapitalManager（经济系统）+ SparseRouter（双通道路由）
│   ├── critic.py        # CriticVC：难度预测 + VC 结算 + 破产重组
│   ├── system.py        # CaMoE_System & CaMoE_Block（v21.1 双通道 + Load Balance）
│   ├── config.py        # v21.1 配置（phase_schedule / economy / route_grad）
│   ├── config_pilot.py  # 0.1B Pilot 配置
│   ├── wrapper.py       # lm-evaluation-harness 适配器
│   └── cuda/            # RWKV-7 ClampW CUDA Kernels（BF16/FP32）
├── scripts/
│   ├── prepare_data.py  # 数据预处理（FineWeb/Cosmopedia tokenize + pack）
│   ├── train_tokenizer.py
│   ├── analyze.py
│   ├── vram_profiler.py
│   └── reset_ckpt.py
├── tokenizer/
│   └── rwkv_vocab_v20230424.txt
├── train.py             # 训练脚本（七阶段 / 路由梯度策略 / Eval / SwanLab）
├── eval.py              # 可视化推理（颜色标注 Trans/RWKV 使用）
├── lmeval.py            # lm-evaluation-harness 评测
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 环境要求
- Python 3.10+
- PyTorch 2.0+（CUDA）
- `pip install -r requirements.txt`

> 注：当前分支包含 `torch.compile` 路径，首次运行会有编译预热时间；若环境不稳定，可临时关闭相关 compile 调用后再训练。

### 数据准备

```bash
# FineWeb-Edu 70% + Cosmopedia-100k 30%
python scripts/prepare_data.py

# 输出：./data/camoe_mix_v20_fineweb70_cosmo30/
# 阶段日志：LOAD+FORMAT → MIX → TOKENIZE+PACK → SAVE
```

💡 **AutoDL 用户**：先执行 `source /etc/network_turbo` 启用学术加速，并设置 `HF_ENDPOINT=https://hf-mirror.com`。

### 训练

```bash
# 0.4B 主力配置
python train.py --scale 0.4b

# 0.4B Toy（快速验证通路，~1000 步）
python train.py --scale 0.4b_toy

# 0.1B Pilot
python train.py --scale 0.1b

# 断点续训
python train.py --scale 0.4b --resume checkpoints/v21.1_0.4b/v21.1_step10000.pth

# 诊断模式
python train.py --scale 0.4b --diag no_amp       # 关闭混合精度
python train.py --scale 0.4b --diag fp32_kernel  # 强制 FP32 CUDA kernel
```

训练脚本会自动：
- 按 `phase_schedule` 切换阶段策略（lr_mult / use_market / route_grad）
- 按 `data_profiles` 切换数据源
- 定期评估验证集 loss 并保存 checkpoint
- 将指标上报 SwanLab（如已安装）

### 推理 / 评估

```bash
# 可视化推理（颜色高亮 Trans 使用情况）
python eval.py

# lm-evaluation-harness 基准评测
python lmeval.py --pretrained checkpoints/v21.1_0.4b/v21.1_final.pth --tasks arc_easy,hellaswag
```

---

## ⚙️ 关键配置项（v21.1）

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `market_alpha_init` | `0.05` | 市场影响力初始值（直接 alpha 值，不是 sigmoid 前 logit） |
| `router_noise_std` | `0.02` | 训练时路由噪声标准差 |
| `aux_loss_coeff` | `0.01` | Load Balance Loss 系数 |
| `route_no_grad` | `True` | 是否对 Critic 路由分支关闭梯度（Gate 分支始终可导） |
| `use_market` | `phase` 级 | 是否启用市场 bias（prewarm 关闭） |
| `route_grad` | `phase` 级 | 是否允许路由梯度（全部开启） |
| `lazy_prefix_union` | `True` | 仅为 Trans 命中 token 构建 prefix |

经济系统参数见 `config.py` 中 `economy` 字典。

---

## 🔮 Roadmap
- **v10–v11**：市场机制、Vickrey 拍卖、Linear-State Bridge
- **v18**：MiniPile-0.4B 主力版本（多阶段训练）
- **v19**：训练稳定性修复（Kernel / Loss / NaN 诊断）
- **v20**：七阶段训练、CriticWarm、经济系统增强
- **v21**：Gradient Gate × Market Bias 双通道路由 + Load Balance Loss
- **v21.1**：`grad_accum` + market 累积更新修复，路由/专家命名澄清
- **v22**：完整 14B token 训练 + Benchmark 验证
- **v23+**：Neurosymbolic Bazaar（Tool-as-Expert + ROSA 记忆专家）

---

## 👥 Contributors
- **S (@shenkunlovecoding) / @艾萨克鸡顿**：Middle School Student / Independent Researcher
  - 架构设计、核心算法、CUDA Kernel、实验设计、数据分析、文档与系统整合

---

## 📝 Citation

```bibtex
@misc{camoe2026,
  author = {S},
  title = {CaMoE: Capital-driven Mixture of Experts with Linear-State Bridges},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shenkunlovecoding/CaMoE}
}
```

---

## 致谢
- 感谢 **Bo Peng** 创造了 RWKV，为线性 Attention 奠定了基础。
- 感谢 **Polymarket** 的预测市场机制带来的灵感。
- 感谢 **TinyStories** 提供的高质量验证数据集。
- 感谢加勒比（我的阿比西尼亚猫）—— 30% 的时候没有他这个项目写不出来，70% 的时候没有它能快 70% 写完。
- 感谢九年义务教育 —— 没有它这个项目不可能存在，但直接导致了这个项目延期了 2 个月。
