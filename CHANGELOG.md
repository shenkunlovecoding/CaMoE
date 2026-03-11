# Changelog

本文档记录 CaMoE v22 之后的重要架构与训练接口变更。

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
