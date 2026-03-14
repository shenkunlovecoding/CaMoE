# Changelog

本文档记录 CaMoE v22 之后的重要架构与训练接口变更。

## v23.0 - 2026-03-14

### Breaking Changes

- 路由语义从 Vickrey 二价拍卖重构为 prediction-market 路由。
- `CriticPair` / REINFORCE 路径被 `RewardCritic` 监督学习路径替代。
- 训练流程从 `prewarm / market_warm / critic_warm / full_market` 四阶段收敛为 `uniform_warmup -> full_market` 两阶段。
- 市场状态不再只看专家资本；每层市场现在显式维护 `wallet(capital)`、`q`、`price` 和共享 `loss_ema`。

### Added

- `wind_rosa/` 与 `Soft_ROSA/` 现已作为 vendored 源码快照直接纳入主仓库管理，不再依赖嵌套 git repo。
- `PredictionMarketRouter`：基于 `shares * (pred_reward - price)` 的 Top-1 路由。
- `MarketStateManager`：管理每层钱包、价格、共享 reward baseline 与 settlement。
- `RewardCritic`：按 token 预测各专家 realized reward 的监督式 critic。
- `market_weight` 与 `exploration_epsilon` 调度，用于从 uniform warmup 平滑切到 full market。
- `tests/test_prediction_market.py`：覆盖价格归一化、结算不变性、reward critic mask、模型集成和 warmup 行为。
- `camoe/rosa_soft_adapter.py` 与 `ROSAExpert(backend in {"wind","soft","sufa","scan"})`，支持本地 `rosa_soft` 路径与 Wind ROSA 并存。
- `camoe/soft_rosa_adapter.py` 与新的实验性 backend：
  - `soft_exact`
  - `soft_qkv1bit`
  - `soft_qkv1bit_reference`
  - `soft_qkv1bit_triton`
  - `soft_qkv1bit_cuda`
- toy 数据扩展：
  - `pattern_complete`
  - `delayed_copy`
  - `first_repeat`
  - `running_max`
  - `sum_threshold`
  - `bracket_depth`
  - `addsub_40`
  - `mixed_v3`

### Changed

- `CaMoE_Block` 的序列市场和 FFN 市场统一到同一套 prediction-market 路由接口。
- 训练脚本与 toy 训练脚本改用 `uniform_warmup`、`market_ramp_steps`、`routing_noise_std`、`exploration_epsilon` 等新配置。
- `market_metrics()` 改为输出 wallet、price、predicted reward、expected profit、exploration 等新诊断指标。
- `eval.py` 的路由统计适配新的 block cache 结构。
- `ROSAExpert` 现在允许直接切到 vendored `Soft_ROSA`：
  - `soft_exact` 走 exact soft DP
  - `soft_qkv1bit*` 在 `rosa_bits == 1` 时提供额外加速路径
- README / README.zh-CN 全面切到 `v23.0` 的 prediction-market 叙述与用法说明。

## v22.2 - 2026-03-14

### Added

- `rosa_soft` SUFA 实验 GPU 快路径：
  - `proxy_triton`
  - `truncated_cuda`
- `rosa_sufa_ops(..., kernel=...)` 调度入口，支持：
  - `auto`
  - `torch`
  - `proxy_triton`
  - `truncated_cuda`
- `CAMOE_ROSA_SUFA_KERNEL` 环境变量，用于在 `backend="sufa"` 下切换实验 kernel。
- `truncated_cuda` 多-spec 进程内编译缓存，支持同一进程内混用多个 `(T, C, K)` 规格。
- `scripts/benchmark_rosa_sufa_kernels.py`，用于对比 `torch / proxy_triton / truncated_cuda` 的 forward 与 step 耗时。
- `tests/test_rosa_sufa_kernels.py`，覆盖：
  - `proxy_triton` 与 torch 参考实现对齐
  - `auto` fallback/warning
  - `truncated_cuda` hard forward 参考正确性
  - 多-spec 单进程 smoke

### Changed

- `backend="sufa"` 默认入口保持不变，但在 CUDA 条件满足时，`kernel="auto"` 会优先尝试 `proxy_triton`。
- `truncated_cuda` 保持为显式 opt-in 实验路径，不会被 `auto` 自动选中。
- `truncated_cuda` backward 回退为现有 SUFA proxy gradient 路径；取消 exact CUDA backward 作为默认实现，因为它没有带来更好的 step-time 收益。

### Notes

- 当前实验路径以 CaMoE 主场景为准：CUDA、`bits_per_symbol <= 8`、`suffix_window <= 8`、`schmitt_trigger == 0`。
- `proxy_triton` 以“保留现有 SUFA 语义”为目标，只替换 proxy 侧窗口展开与衰减内核。
- `truncated_cuda` 以“更激进的 forward 提速”为目标，hard forward 语义为截断版本，训练时仍通过 SUFA proxy backward 近似求梯度。

## v22.1 - 2026-03-09

### Breaking Changes

- 序列市场重构为 `TimeMix vs ROSA`，ROSA 不再与 FFN 专家同池竞争。
- `ROSAExpert` 切换到 vendored `Wind ROSA` 实现，不再使用 `rosa_soft` 路径。
- ROSA 配置字段重命名：
  - `rosa_n_bits` -> `rosa_bits`
  - `rosa_n_heads` -> `slim_rosa_heads`
  - `rosa_suffix_window` -> `rosa_truncation_length`
- 删除 shared `DeepEmbed` 输入旁路，DeepEmbed 只保留为 FFN 市场专家。
- 旧 checkpoint 不保证可直接 `strict=True` 加载。

### Added

- 双市场架构：
  - `Sequence Market`: `TimeMixExpert` vs `ROSAExpert`
  - `FFN Market`: `RWKVExpert` / `DeepEmbedExpert` / `SlimDeepEmbedExpert`
- `Wind ROSA` 本地 adapter 与 vendored kernel 编译加载链路。
- Slim Wind ROSA 语义：
  - matched `1 -> +e`
  - matched `0 -> -e`
  - unmatched `-> 0`
- `DeepEmbedExpert`
  - `1x / 4x` 两种模式
  - 支持基于 `token_ids` 的稀疏 dispatch
- `SlimDeepEmbedExpert`
  - 低秩调制版 DeepEmbed
  - 作为轻量 FFN 市场专家
- `FractalCaMoEPlaceholder`
  - 分形子专家占位符
  - `ExpansionRightLedger`
  - 支持“购买展开权”但默认不接入主训练路径
- reverse-digits toy 实验扩展：
  - `reverse_digits`
  - `copy_digits`
  - `mixed_digits`
  - `parity_digits`
  - `cumsum_mod10`
  - `majority_vote`
  - `count_ones`
  - `mixed_all_digits`
- baseline 扩展：
  - `single_rosa`
  - `single_rwkv`
  - `camoe`

### Changed

- FFN 市场从同构 `RWKVExpert` 池扩展为异构专家池。
- `n_experts` 现在表示 RWKV FFN 专家数量。
- 新增 `total_ffn_experts` 作为 FFN 市场总专家数。
- `token_ids` 现在会跟随 sparse dispatch 传入 DeepEmbed 类专家。
- 资本系统继续使用比例折旧，并加入资本上限。
- Critic reward 结算支持 `critic_profit_clip`。
- 市场诊断缓存统一包含：
  - `winners`
  - `prices`
  - `bids`
  - `positions`
  - `expert_capitals`
- toy 实验日志新增：
  - 分任务评估
  - 分任务路由统计
  - ANSI / HTML 路由可视化

### Removed

- `SharedDeepEmbed`
- `use_deep_embed`
- `deep_embed_scale`
- `rosa_soft` 相关后端分支
- Top-K / weight mixing 兼容逻辑，当前为严格 Top-1 winner-takes-all

### Tooling

- `scripts/dump_camoe_text.py` 已更新为导出 v22.1 核心文件：
  - 双市场模型
  - DeepEmbed 专家
  - Fractal placeholder
  - Wind ROSA adapter / kernel
- 根目录 `llm.txt` 已按新导出清单重生成。

### Notes

- 当前 Windows + CUDA 环境下，Wind ROSA 原生 kernel 已可编译和前后向运行。
- `torch.utils.cpp_extension` 仍可能打印 `cl` 版本读取编码 warning，但不影响实际编译与执行。
