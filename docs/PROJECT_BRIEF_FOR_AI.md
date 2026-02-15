# CaMoE 项目说明（给其他 AI）

## 1. 项目定位
CaMoE（Capital-driven Mixture of Experts）是一个混合专家语言模型项目。
核心思路是把传统 MoE 的静态门控改为“市场机制路由”：
- 专家按 token 进行竞价（Vickrey/第三价清算）
- 专家资本动态更新（收益、税收、保底）
- Critic 预测难度并对报价提供补贴/惩罚

当前主线版本是 v18.5-test，默认规模 0.4b。

## 2. 核心架构（从上到下）
### 2.1 系统主干
- `CaMoE/system.py`
  - `CaMoE_System`：整网入口，负责 embedding、多层 block、输出 head、市场状态维护。
  - `CaMoE_Block`：单层结构，顺序为：
    1) RWKV7 TimeMix
    2) 可选 DEA 分支（DeepEmbedAttention）
    3) Critic + Router 做 Top-2 专家选择
    4) 专家执行并加权融合

### 2.2 Backbone 与 DEA
- `CaMoE/backbone.py`
  - `RWKV7_TimeMix`：RWKV 主干（依赖自定义 CUDA 算子）
  - `DeepEmbedAttention`：并行注意力分支
  - `SharedDeepEmbed`：跨层共享的 token embedding 调制表（降低参数和显存）

当前 DEA 设计要点：
- 低维路径：`q_dim=256`，`kv_dim=32`
- K/V 由 token 级 DeepEmbed 调制
- 与 TimeMix 使用同一份 pre-norm 输入并行计算

### 2.3 专家与路由
- `CaMoE/experts.py`
  - `SparseRWKVFFN`：RWKV 风格 FFN 专家
  - `LinearTransformerExpert`：使用 bridge prefix 的 transformer 专家
- `CaMoE/market.py`
  - `SparseRouter`：Top-2 选举 + 第三价清算
  - `CapitalManager`：资本更新与健康约束
- `CaMoE/critic.py`
  - `CriticVC`：difficulty + affinity（做多/做空）

### 2.4 Bridge
- `CaMoE/bridge.py`
  - `UltimateBridge`：把 RWKV 状态映射为 transformer 专家使用的 prefix

## 3. 配置与运行入口
### 3.1 配置
- `CaMoE/config.py`
  - 默认版本：`v18.5-test`
  - 默认规模：`0.4b`
  - 关键字段：
    - `num_rwkv_experts`, `num_trans_experts`, `top_k`
    - `use_deep_embed_attention`, `use_shared_deep_embed`
    - `dea_q_dim`, `dea_kv_dim`
    - `data_path`, `data_roots`, `mix`

### 3.2 训练
- `train.py`
  - 支持断点续训、阶段训练（prewarm/warmup/normal）、验证损失评估
  - prewarm 阶段当前会训练：
    - transformer experts
    - bridge / critic / capital
    - `dea` 与 `deep_embed` 参数

### 3.3 评估
- `eval.py`：交互式/可视化文本生成，输出每层 transformer 使用比例
- `lmeval.py` + `CaMoE/wrapper.py`：对接 lm-evaluation-harness

## 4. 代码阅读建议（给 AI）
如果你要快速理解行为，不要从 README 开始，建议按这个顺序：
1. `CaMoE/config.py`（先看开关和规模）
2. `CaMoE/system.py`（全链路）
3. `CaMoE/backbone.py`（RWKV 与 DEA 细节）
4. `CaMoE/market.py` 与 `CaMoE/critic.py`（路由经济机制）
5. `train.py`（训练阶段控制、checkpoint 逻辑）

## 5. 关键行为假设
- 路由是 token 级别的 Top-2 混合，不是整层固定专家。
- 市场机制默认全程开启（即便 prewarm/warmup 也路由）。
- RWKV CUDA kernel 可用性会影响速度与可运行性。
- 推理和评估脚本默认按 checkpoint 内 config 对齐模型结构。

## 6. 目录速览
- `CaMoE/`：模型核心代码
- `train.py`：训练入口
- `eval.py`：可视化生成
- `lmeval.py`：基准评测入口
- `scripts/`：数据准备、分析和工具脚本
- `tokenizer/`：RWKV tokenizer 相关文件
- `model/`、`checkpoints/`：底模和训练权重（通常较大）

## 7. 与其他模型仓库的不同点
- 不是标准 dense transformer 训练逻辑
- 不是传统 aux-loss 负载均衡 MoE
- 路由决策显式耦合资本与难度建模
- 支持 RWKV + Transformer 的混合协作

## 8. 如果你要继续改这个项目
优先保持以下不变量：
1. 输入输出 tensor 形状不变
2. Router 返回四元组语义不变（winners/weights/price/bids）
3. checkpoint 加载路径和字段兼容（`model`, `optimizer`, `config`, `step`）
4. config 开关有默认值，旧 checkpoint 不应直接崩溃
