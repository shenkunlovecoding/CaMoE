# CaMoE 项目更新报告（Update）

更新日期：2026-02-19  
项目路径：`g:\DG-D\CaMoE_Project`

## 1. 本次更新摘要

本次 Report 相比上一版，新增了两类信息：
- 新增「提交演进时间线」：覆盖 2026-02-03 至 2026-02-19 的版本演进与关键节点。
- 新增「现阶段判断」：结合当前代码主线（`v20`）与最近提交（`prepare_data` / `critic warm`）给出项目状态评估。

当前结论：项目已从 v18 稳定化、v19 过渡，进入 v20 的训练策略与数据配方强化期，工程重心从“能跑”转向“训练经济系统可控 + 数据配方有效”。

---

## 2. 提交演进时间线（按阶段）

### 阶段 A：初始搭建与早期验证（v11 -> v12）
时间：2026-02-03

关键提交：
- `1151e09` Initial commit: CaMoE v11.0 Release
- `fa14b60` v12
- `85fa873` clean,ver12.0
- `44d5598` 修复“强行覆盖参数”与 eval

阶段特征：
- 快速迭代、边开发边修复。
- 重点在基础训练闭环与评测链路可用性。

### 阶段 B：结构扩展（v15 -> v16）
时间：2026-02-04 ~ 2026-02-05

关键提交：
- `1053a89` v15
- `4384a8f` v16

阶段特征：
- 进入架构演进周期，功能扩展快于文档沉淀。

### 阶段 C：v18 稳定化与分支融合（v18 / v18.5）
时间：2026-02-06 ~ 2026-02-16

关键提交：
- `03a8a6d` first v18 test
- `1142763` update v18
- `3fb83f0` v18 stable
- `edad263` update to v18 stable version
- `db247b8` v18.5-test
- `9fac12c` v18.5.1
- `94b8374` merge v18.5-test
- `a640ad5` cuda
- `d6402fc` / `6fcd039` fix
- `d178cc3` config
- `c362007` docs: README 稳定性与 kernel 兼容性说明

阶段特征：
- 明显聚焦训练稳定性与 CUDA 兼容性。
- 文档开始同步工程问题与修复路径。

### 阶段 D：v19 -> v20 过渡（经济系统与调度增强）
时间：2026-02-16 ~ 2026-02-19

关键提交：
- `db69caa` v19
- `1cfdfc9` version number
- `56b7064` data
- `a8ac6b6` v20
- `b3d425a` a little change
- `55734ba` critic warm
- `63ca422` prepare_data

阶段特征：
- 训练流程从单一阶段走向多阶段策略。
- 经济系统与 Critic 训练被单独强化（`critic warm`）。
- 数据侧投入加大（`data`、`prepare_data`）。

---

## 3. 当前代码主线（与提交历史对应）

### 3.1 配置主版本：v20
- `CaMoE/config.py` 中主版本为 `VERSION = "v20"`。
- 使用七阶段训练调度（`prewarm / warm / criticwarm / prenormal / normal / sft / rlhf`）。
- 数据 profile 采用可切换混配方式（FineWeb/Cosmopedia/UltraChat 等）。

### 3.2 系统架构
- 主系统：`CaMoE/system.py`（`CaMoE_System` + `CaMoE_Block`）。
- 路由：`CaMoE/market.py`（Top-2 Vickrey 风格路由 + 资本管理）。
- Critic：`CaMoE/critic.py`（难度建模、affinity 补贴、结算、债务/重组机制）。
- Backbone：`CaMoE/backbone.py`（RWKV7 TimeMix + CUDA 扩展 + fallback 路径）。
- Bridge/Experts：`CaMoE/bridge.py`、`CaMoE/experts.py`。

### 3.3 训练与评测链路
- 训练入口：`train.py`
- 可视化生成评测：`eval.py`
- 基准评测入口：`lmeval.py` + `CaMoE/wrapper.py`
- 数据预处理：`scripts/prepare_data.py`（近期提交重点）

---

## 4. 对最近两次关键提交的解释

### `55734ba` critic warm
含义与作用：
- 表示训练策略将 Critic 的学习单独拉出一个阶段（或强化其阶段），提升路由经济系统中的“价格信号质量”。
- 预期收益：
  - 更早建立稳定的 difficulty/affinity 估计；
  - 减少专家竞争中的噪声选择；
  - 提高后续 normal 阶段市场更新的可解释性。

### `63ca422` prepare_data
含义与作用：
- 表示工程重心转向“数据质量与配方一致性”。
- 预期收益：
  - 训练损失曲线更稳定；
  - 不同阶段 profile 的切换更可控；
  - 下游评测波动降低（尤其是常识/阅读理解类任务）。

---

## 5. 项目现状评估

### 5.1 成熟度判断
- 架构成熟度：中高（核心模块齐全，机制完整）。
- 训练工程成熟度：中高（断点续训、阶段调度、诊断开关完整）。
- 数据工程成熟度：中（正在快速强化，近期提交集中在该方向）。
- 文档一致性：中（历史版本叠加较多，建议持续做“主线化”整理）。

### 5.2 当前优势
- 机制创新清晰：经济路由 + Critic 结算不是概念，代码已落地。
- 训练可控性提升：多阶段与参数分组能针对子系统定向优化。
- 诊断能力较强：对 NaN、AMP、CUDA kernel 兼容问题有明确开关与 fallback。

### 5.3 当前风险
- 版本迭代很快，commit message 偏短（如 `fix` / `a little change`），后期审计成本上升。
- 数据配方仍在演进，实验可重复性依赖配置快照与数据版本锁定。
- README 与代码可能存在阶段性口径差（常见于快速迭代期）。

---

## 6. 建议的下一步（面向下一版 Report）

1. 建立最小实验记录模板（每次训练保存：配置哈希、数据配方、起始 checkpoint、评测任务）。
2. 统一 commit 规范（建议 `scope: summary`），把 `fix` 改为可检索语义。
3. 增加 `CHANGELOG.md`，按 v19/v20 聚合重要行为变化，减轻 README 负担。
4. 给 `scripts/prepare_data.py` 增加数据统计产物（token 分布、样本长度分位数、清洗过滤率）。

---

## 7. 一句话结论

截至 2026-02-19，CaMoE 已进入 v20 的“训练策略与数据工程协同优化”阶段，核心创新点已从原型验证迈向工程化打磨，下一步的关键是实验可重复性与版本叙事统一。
