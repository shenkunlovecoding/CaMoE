## CaMoE: Capital-driven Mixture of Experts

> "We choose to go to the moon... not because they are easy, but because they are hard."

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Model Architecture](https://img.shields.io/badge/Architecture-Hybrid_MoE-blueviolet)](https://github.com/shenkunlovecoding/CaMoE)
[![Speed](https://img.shields.io/badge/Speed-7k_TPS-orange)](https://github.com/shenkunlovecoding/CaMoE)

**CaMoE (Capital-driven Mixture of Experts)** 是一个基于**市场经济机制**的混合专家语言模型架构，目前主力版本为 **v18 · MiniPile-0.4B · 6R2T-Top2**。

不同于传统 MoE 依赖静态门控或辅助损失（Auxiliary Loss），CaMoE 引入了 **Vickrey 拍卖**、**资本动态** 和 **做空机制**，让 RWKV（线性状态机）和 Transformer（注意力机制）专家通过自由市场竞争实现算力的自然分工。

## 🏆 Benchmark（历史 0.1B TinyStories 实验）

在 TinyStories 数据集上，早期 0.1B 规模的 CaMoE 展示了良好的收敛速度和推理效率（供参考，非 v18 主力实验）：

| Metric | Result | Note |
| :--- | :--- | :--- |
| **PPL (Perplexity)** | **2.16** | 逻辑连贯，语法完美 |
| **Speed (Inference)** | **24,918 TPS** | 比同级 Dense Transformer 快 **5-8倍** |
| **Avg Trans Usage** | **12.59%** | 实现了深度稀疏化 (Depth-wise Sparsity) |

### 🧠 涌现的层级分工 (Emergent Specialization)
CaMoE 在没有人工干预的情况下，自动学会了在不同层级分配不同的算力（示例自 0.1B TinyStories 实验）：
```
 L00-L04 | 🟦 RWKV Dominant  | 基础词法与浅层语义 (省钱模式)
 L05     | 🟥 Trans Dominant | 逻辑中枢与复杂推理 (逻辑转折点)
 L06-L09 | 🟦 RWKV Dominant  | 信息传递与上下文维持
 L10-L11 | 🟥 Trans Dominant | 输出精修与Token选择 (最终把关)
```

## 🌟 核心创新

### 1. Market Mechanism (市场机制)
- **Vickrey 拍卖**：采用第二价格拍卖，激励专家诚实报价（Truthful Bidding）。
- **资本动态 & 累进税**：实现优胜劣汰，同时防止马太效应（Winner-takes-all），无需 Auxiliary Loss 即可实现负载均衡。

### 2. Linear-State Bridge (线性状态桥)
- **Low-Rank Projection**：将 RWKV 的 RNN Hidden State 通过低秩投影传递给 Transformer。
- **O(1) Complexity**：Transformer 专家被设计为 Linear Cross-Attention，不回溯历史 KV Cache，使得整个系统保持线性推理复杂度。

### 3. Critic as VC (风投模式)
- Critic 不仅预测难度，还作为做市商 (Market Maker)。
- 支持 **做多 (Long)** 和 **做空 (Short)**：如果专家过度自信但答错，Critic 会通过做空剥夺其资本。

## 📁 项目结构（v18）

```text
CaMoE_Project/
├── CaMoE/
│   ├── backbone.py      # RWKV-7 TimeMix 主干 + CUDA Kernel 封装
│   ├── bridge.py        # UltimateBridge：低秩 Linear-State Bridge
│   ├── experts.py       # Sparse RWKV & Linear Transformer 专家
│   ├── market.py        # CapitalManager + SparseRouter（Vickrey 拍卖）
│   ├── critic.py        # CriticVC：难度预测 + 专家偏好建模
│   ├── system.py        # CaMoE_System & CaMoE_Block 主架构
│   ├── config.py        # v18 / MiniPile 配置（0.4b & 0.1b）
│   ├── config_pilot.py  # 小规模 Pilot 配置（快速实验）
│   ├── wrapper.py       # CaMoELM：lm-evaluation-harness 适配器
│   └── cuda/            # RWKV-7 自定义 CUDA Kernels
├── scripts/
│   ├── prepare_data.py  # 使用 RWKV tokenizer 预处理数据（如 MiniPile）
│   ├── train_tokenizer.py
│   ├── analyze.py
│   ├── vram_profiler.py
│   └── reset_ckpt.py
├── tokenizer/
│   ├── rwkv_tokenizer.py
│   └── rwkv_vocab_v20230424.txt
├── train.py             # v18 训练脚本（支持断点续训 / Eval Loss）
├── eval.py              # 可视化单样本评测（颜色标注 Trans/RWKV 使用）
├── lmeval.py            # 接入 lm-evaluation-harness 的评测脚本
├── requirements.txt
└── README.md
```

## 🚀 Quick Start（v18）

### 环境要求
- Python 3.10+
- PyTorch 2.0+（推荐 CUDA 版本）
- 安装依赖：

```bash
pip install -r requirements.txt
```

### 数据准备（以 MiniPile 为例）

1. 按需下载/准备 MiniPile 等数据集，并放入 `data/` 目录。
2. 使用脚本预处理为训练格式（根据你自己的数据路径适当修改脚本参数）：

```bash
python scripts/prepare_data.py
```

确保 `CaMoE/config.py` 中的 `data_path` 等路径与你实际的数据位置一致（v18 默认使用 `./data/minipile_processed`）。

### 训练（v18 主力：MiniPile-0.4B · 6R2T-Top2）

1. 打开 `CaMoE/config.py`，确认/修改以下关键字段：
   - `VERSION = "v18"`
   - `SCALE = "0.4b"` 或 `"0.1b"`
   - `data_path` / `weights_path` / `save_dir` 等路径

2. 启动训练：

```bash
# 使用 0.4B 主力配置（v18）
python train.py --scale 0.4b

# 使用 0.1B Pilot 配置（快速实验）
python train.py --scale 0.1b

# 从已有 checkpoint 继续训练
python train.py --scale 0.4b --resume path/to/your_checkpoint.pth
```

如果 `--resume` 未指定且在 `checkpoints/v18_0.4b/init.pth` 存在初始化权重，训练脚本会自动使用该权重作为起点；否则会从 `config['weights_path']` 中的 RWKV 底模加载。

### 推理 / 评估

#### 1. 可视化单样本评测（颜色高亮 Trans 使用情况）

在 `eval.py` 中设置好 `MODEL_PATH`、`SCALE` 等参数后运行：

```bash
python eval.py
```

脚本会对若干示例 prompt 生成文本，并用不同颜色标注哪些 token 更依赖 Transformer 专家，同时打印每层 Transformer 使用比例。

#### 2. 基准评测（lm-evaluation-harness）

`lmeval.py` 会从 checkpoint 中读取 `config` 以匹配当前架构（0.1b/0.4b）；未提供 checkpoint 时用 `--scale` 选择配置。结果 JSON 若不指定 `--output` 则自动命名为 `results_{version}_{scale}_{tasks}_{timestamp}.json`。

```bash
# 使用 v18 checkpoint，自动匹配架构并生成结果文件名
python lmeval.py --pretrained checkpoints/v18_0.4b/v18_step2000.pth --tasks arc_easy,hellaswag

# 仅指定规模（随机初始化或你自行加载权重时）
python lmeval.py --scale 0.4b --tasks arc_easy --output my_results.json
```

## 🔮 Roadmap（简版）
- [x] **v10–v11**：市场机制、Vickrey 拍卖、Linear-State Bridge、显存优化与断点续训验证
- [x] **v18**：MiniPile-0.4B · 6R2T-Top2 主力版本（长预热、多阶段训练、自动市场路由）
- [ ] **v19**: DeepEmbed 集成 (参数稀疏化) & Fused Kernel (算子融合加速)。
- [ ] **v20**: Neurosymbolic Bazaar。引入 Tool-as-Expert (计算器/搜索) 和 ROSA (记忆网络) 专家。

## 👥 Contributors
- **S (@shenkunlovecoding) / @艾萨克鸡顿**：Middle School Student / Independent Researcher
架构设计、核心算法、CUDA Kernel、实验设计、数据分析、文档与系统整合

## 📝 Citation
如果你觉得这个项目有启发，请引用我们的工作：

```bibtex
@misc{camoe2026,
  author = {S},
  title = {CaMoE: Capital-driven Mixture of Experts with Linear-State Bridges},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shenkunlovecoding/CaMoE}
}
```

## 致谢
- 感谢 Bo Peng 创造了 RWKV，为线性 Attention 奠定了基础。
- 感谢 Polymarket 的预测市场机制带来的灵感。
- 感谢 TinyStories 提供的高质量验证数据集。
- 感谢 加勒比 我的阿比西尼亚猫，30%的时候没有他这个项目写不出来，70%的时候这个项目没有它能快70%写完
- 感谢 九年义务教育 没有它这个项目不可能存在，但直接导致了这个项目延期了2个月