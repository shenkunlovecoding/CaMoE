# CaMoE: Capital-driven Mixture of Experts

> "We choose to go to the moon... not because they are easy, but because they are hard."

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Model Architecture](https://img.shields.io/badge/Architecture-Hybrid_MoE-blueviolet)](https://github.com/shenkunlovecoding/CaMoE)
[![Speed](https://img.shields.io/badge/Speed-25k_TPS-orange)](https://github.com/shenkunlovecoding/CaMoE)

**CaMoE (Capital-driven Mixture of Experts)** 是一个基于**市场经济机制**的混合专家语言模型架构。

不同于传统 MoE 依赖静态门控或辅助损失（Auxiliary Loss），CaMoE 引入了 **Vickrey 拍卖**、**资本动态** 和 **做空机制**，让 RWKV（线性状态机）和 Transformer（注意力机制）专家通过自由市场竞争实现算力的自然分工。

## 🏆 Benchmark Result (0.1B Scale)

在 TinyStories 数据集上，CaMoE 展示了惊人的收敛速度和推理效率。

| Metric | Result | Note |
| :--- | :--- | :--- |
| **PPL (Perplexity)** | **16.20** | 逻辑连贯，语法完美 |
| **Speed (Inference)** | **24,918 TPS** | 比同级 Dense Transformer 快 **5-8倍** |
| **Avg Trans Usage** | **12.59%** | 实现了深度稀疏化 (Depth-wise Sparsity) |

### 🧠 涌现的层级分工 (Emergent Specialization)
CaMoE 在没有人工干预的情况下，自动学会了在不同层级分配不同的算力：
```
 L00-L04 | 🟦 RWKV Dominant  | 基础词法与浅层语义 (省钱模式)
 L05     | 🟥 Trans Dominant | 逻辑中枢与复杂推理 (逻辑转折点)
 L06-L09 | 🟦 RWKV Dominant  | 信息传递与上下文维持
 L10-L11 | 🟥 Trans Dominant | 输出精修与Token选择 (最终把关)
```

# 🌟 核心创新

## 1. Market Mechanism (市场机制)
- **Vickrey 拍卖**：采用第二价格拍卖，激励专家诚实报价（Truthful Bidding）。
- **资本动态 & 累进税**：实现优胜劣汰，同时防止马太效应（Winner-takes-all），无需 Auxiliary Loss 即可实现负载均衡。

## 2. Linear-State Bridge (线性状态桥)
- **Low-Rank Projection**：将 RWKV 的 RNN Hidden State 通过低秩投影传递给 Transformer。
- **O(1) Complexity**：Transformer 专家被设计为 Linear Cross-Attention，不回溯历史 KV Cache，使得整个系统保持线性推理复杂度。

## 3. Critic as VC (风投模式)
- Critic 不仅预测难度，还作为做市商 (Market Maker)。
- 支持 **做多 (Long)** 和 **做空 (Short)**：如果专家过度自信但答错，Critic 会通过做空剥夺其资本。

# 📁 项目结构

```
CaMoE/
├── backbone.py    # RWKV-7 Linear Backbone
├── bridge.py      # Low-Rank Linear-State Bridge
├── experts.py     # Sparse RWKV & Linear Transformer Experts
├── market.py      # 资本管理、拍卖与路由逻辑
├── camoe.py       # 系统主架构
├── train.py       # 训练脚本 (支持 BF16, Checkpointing,Resume)
├── generate.py    # 推理生成脚本
├── config.py      # 训练配置
└── cuda/          # RWKV CUDA Kernels
```

# 🚀 Quick Start

## 环境要求
- Python 3.10+
- PyTorch 2.0+ (推荐 CUDA 版本)
- `pip install -r requirements.txt`

## 训练
先修改config.py确保各类路径正确
```bash
# 自动加载数据并开始训练
python train.py --resume '你的checkpoint'
```

## 评估
```bash
python eval.py #单样本详细测试
python benchmark.py #多样本平均测试
```

# 🔮 Roadmap
- [x] **v10.0**：市场机制、Vickrey 拍卖、混合架构验证 (Completed)
- [x] **v11.0**：Linear-State Bridge、显存优化、断点续训 (Completed)
- [ ] **v12.0**：Dream Mode (睡眠时自动优化整理记忆)
- [ ] **v13.0**：Tool as Expert (将计算器、搜索引擎封装为专家参与拍卖)

# 👥 Contributors
- **S (@shenkunlovecoding)**：架构设计、核心算法、CUDA Kernel、实验设计、数据分析、文档与系统整合

# 📝 Citation
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

# 致谢
- 感谢 Bo Peng 创造了 RWKV，为线性 Attention 奠定了基础。
- 感谢 Polymarket 的预测市场机制带来的灵感。
- 感谢 TinyStories 提供的高质量验证数据集。
```