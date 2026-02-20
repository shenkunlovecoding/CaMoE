# ROSA 分支 vs 主分支（详细版）

本文对比当前 `exp/rosa-proto` 与 `master` 的核心差异，并说明 ROSA 的概念与数学原理。

## 1. ROSA 是什么

ROSA（这里指当前实现的 `1-bit ROSA`）可以理解为一条“离散序列模式分支”：

- 先把连续隐藏状态 `h_t` 投影成多个二值流（bit streams）
- 在每条 bit 流上做在线模式匹配（基于后缀自动机思想）
- 产出一个“下一符号建议”序列（找不到则输出 no-match）
- 再把这些离散结果嵌入回连续向量，作为额外特征注入主干

它与标准注意力/TimeMix不同：不是直接做实值相似度加权，而是显式利用离散重复结构（repeat / motif / suffix reuse）。

## 2. ROSA 的数学直觉与原理

### 2.1 输入到二值流

对某层输入 `x in R^{B x T x C}`，先线性投影到 `S` 条流：

- `z_{b,t,s} = W_s x_{b,t}`
- `bit_{b,t,s} = 1[z_{b,t,s} > 0]`

所以每条流是长度 `T` 的二值序列 `x^(s) = (x_1, ..., x_T), x_i in {0,1}`。

### 2.2 在线后缀自动机（SAM）式状态扩展

在 `rosa_1bit_sequence` 中，对每个时间步 `i` 在线扩展自动机状态，维护：

- 转移表 `b`
- 后缀链接 `c`
- 状态长度 `d`
- 最新结束位置 `e`

这等价于维护“历史所有子串的紧凑表示”，支持快速回溯“当前上下文在历史中最像谁”。

### 2.3 预测规则（核心）

对当前位置 `i`，沿后缀链找第一个有效状态 `v`，取其历史结束位置 `e[v]`，然后读历史中的“后继符号”：

- `nxt = e[v] + 1`
- 若 `nxt < T`，输出 `y_i = x_{nxt}`
- 否则 `y_i = -1`（no-match）

可理解为：

- “找到历史上最相关的一段上下文”
- “用那段上下文后面出现过的符号当作当前建议”

这是一种离散版的 next-symbol retrieval。

### 2.4 回到连续空间

代码里将 `-1` 映射到类别 `2`，所以 token 集合是 `{0,1,2}`：

- `e_i = Emb(y_i')`, `y_i' in {0,1,2}`
- 多流平均池化：`p_t = mean_s e_{t,s}`
- 线性映射回模型维度：`r_t = W_o p_t`
- 归一化：`rosa_out_t = LN(r_t)`

最终在 block 中残差注入：

- 主分支：`x + att_out (+ dea_out)`
- ROSA 分支：`x + att_out (+ dea_out) + rosa_out`

## 3. 当前代码中的 ROSA 实现映射

- 核心文件：`camoe/rosa.py`
  - `rosa_1bit_sequence`: 在线 SAM 风格 1-bit 变换
  - `ROSA1bitLayer`: `to_bits -> ROSA -> Embedding -> out -> norm`
- 系统接入：`camoe/system.py`
  - `CaMoE_Block.__init__` 按配置创建 `self.rosa`
  - `_forward_att_stage` 把 `rosa_out` 加到残差
- 配置开关：
  - `camoe/config.py`
  - `camoe/config_pilot.py`
  - 字段：`use_rosa`, `rosa_num_streams`, `rosa_emb_dim`

## 4. ROSA 分支相对主分支的工程差异

### 4.1 新增模块与开关

- 新增：`camoe/rosa.py`
- 新增配置：
  - `use_rosa: bool`（默认 `False`）
  - `rosa_num_streams: int`（默认 `32`）
  - `rosa_emb_dim: int`（默认 `64`）

### 4.2 训练分组策略

- 文件：`train.py`
- 在 `_classify_param_group` 中：
  - `if ".rosa." in name: return "bridge"`
- 含义：
  - ROSA 参数归到 `bridge` 组
  - prewarm 阶段可以随 `bridge` 一起训练，保持 ROSA 早期行为一致

### 4.3 Backbone 差异（与 master）

- 文件：`camoe/backbone.py`
- 当前 ROSA 分支里，backbone CUDA路径更精简（比 master 少一些调试/回退开关链）
- 但已包含 state 小改动：
  - `RUN_CUDA_RWKV7` 返回 `(x_att, sa)`
  - `state_representation = ln_sa(sa)`

## 5. 理论优点与现实约束

### 5.1 可能收益

- 对重复结构、模板化上下文更敏感
- 为主干提供与注意力互补的离散先验
- 可作为可开关的并行增强分支，便于做 ablation

### 5.2 当前实现约束

- `rosa_1bit_sequence` 在 Python 循环中逐 batch、逐 stream 执行，开销较高
- bit 阈值化是硬离散操作，梯度主要经过后续 embedding/out，不会穿过离散算法本体
- 因为是实验分支，建议默认关，按实验计划开启

## 6. 快速结论

ROSA 分支本质是在 v21 主干上增加一条“离散序列模式检索”并行特征通道。  
它不替代原有 TimeMix/路由系统，而是补充另一类结构信号；工程上通过 `use_rosa` 可完全开关，便于独立评估收益与成本。
