# CaMoE 数据集挑选任务说明（给其他 AI）

> 目标：你不是来改代码的。你要为 CaMoE 训练方案挑选和组合数据集，并给出可执行的配比与理由。

## 0. 项目背景（你必须先理解）
CaMoE 是 RWKV + Transformer 的混合专家模型，当前主线是 `v18.5-test`（默认 `0.4b`）。
训练入口 `train.py` 支持多数据集混合（`interleave_datasets`），配置在 `CaMoE/config.py`。

当前 `0.4b` 配置中，数据相关字段是：
```python
"data_roots": {
    "tinystories": "./data/tinystories_rwkv_processed",
    "dialog": "./data/dailydialog_rwkv_processed",
    "chat": "./data/ultrachat_rwkv_processed",
    "minipile": "./data/minipile_rwkv_processed",
},
"mix": {
    "tinystories": 0.6,
    "dialog": 0.2,
    "chat" : 0.2,
    "minipile": 0.0,
},
```

你要做的是：替用户挑选更好的训练数据组合，不写训练算法，不改模型结构。

---

## 1. 你的任务边界
### 1.1 你必须完成
1. 给出候选数据集清单（>= 10 个，按用途分组）
2. 给出筛选结果（最终入选 3-6 个）
3. 给出配比方案（至少 2 套：保守版 / 激进版）
4. 给出风险与回退策略
5. 给出落地建议（如何映射到现有 `data_roots` 和 `mix`）

### 1.2 你禁止做
1. 不要改模型代码
2. 不要改路由机制、损失函数
3. 不要给“泛泛建议”而不落到具体数据集名称、规模和许可证

---

## 2. 硬约束（必须满足）
1. **许可证可商用或至少研究可用**，并注明限制条款
2. **可获得性稳定**（HF 可下载、官方镜像可访问）
3. **文本质量可控**（噪声、重复、乱码比例可估）
4. **语言分布符合目标**（当前项目偏英文通用能力）
5. **能转成 `input_ids` 长序列**（兼容 RWKV tokenizer 管线）
6. **成本可承受**（下载量、处理时长、磁盘占用）

---

## 3. 评分标准（你要按这个打分）
对每个候选数据集按 1-5 分打分，给总分（加权）：

- `quality`（25%）：文本质量、重复率、脏数据比例
- `coverage`（20%）：知识覆盖、领域广度
- `reasoning_signal`（20%）：推理/结构化语料密度（对话、问答、解释）
- `license_safety`（15%）：许可证清晰度
- `cost_efficiency`（10%）：下载/清洗成本
- `format_fitness`（10%）：与当前预处理脚本兼容性

总分公式：
```text
total = 0.25*quality + 0.20*coverage + 0.20*reasoning_signal + 0.15*license_safety + 0.10*cost_efficiency + 0.10*format_fitness
```

---

## 4. 推荐候选池（先从这里评估）
> 你可以扩展，但至少覆盖以下类别。

### 4.1 通用预训练文本（base knowledge）
1. `monology/pile-uncopyrighted`
2. `allenai/c4`（可选精简子集）
3. `HuggingFaceFW/fineweb`（按 token 预算抽样）
4. `togethercomputer/RedPajama-Data-1T`（注意子集选择）

### 4.2 叙事与语言流畅度（style/fluency）
1. `roneneldan/TinyStories`
2. `HuggingFaceTB/cosmopedia-100k`

### 4.3 对话与指令（assistant behavior）
1. `openbmb/ultrachat`
2. `lmsys/lmsys-chat-1m`（若许可证与可得性合适）
3. `Open-Orca/OpenOrca`（核查许可证）
4. `Anthropic/hh-rlhf`（仅在许可证允许条件下）

### 4.4 高质量学术/知识型（可选）
1. `allenai/dolma`（抽样）
2. `wikimedia/wikipedia`（英文）

---

## 5. 你要输出的最终结果格式（严格按这个给我）

### 5.1 候选评分表
```markdown
| Dataset | Type | Size(rough) | License | quality | coverage | reasoning_signal | license_safety | cost_efficiency | format_fitness | total | Keep? |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
```

### 5.2 最终入选 + 配比
```markdown
方案A（保守）:
- dataset_a: 0.xx
- dataset_b: 0.xx
- ...
理由: ...

方案B（激进）:
- dataset_a: 0.xx
- dataset_b: 0.xx
- ...
理由: ...
```

### 5.3 风险清单
```markdown
- 风险1: ...
  - 缓解: ...
- 风险2: ...
  - 缓解: ...
```

### 5.4 直接可落地配置片段
输出可直接贴进 `CaMoE/config.py` 的 `data_roots` 与 `mix` 示例。

---

## 6. 与本项目代码的对接方式

### 6.1 预处理脚本能力
项目已有 `scripts/prepare_data.py`，目前支持：
- raw 文本源
- chat 对话源
- 多源 interleave
- RWKV tokenizer 打包

核心配方结构：
```python
DATA_RECIPE = {
    "tinystories": ("roneneldan/TinyStories", "train[:10%]", "raw", 0.4),
    "cosmopedia":  ("HuggingFaceTB/cosmopedia-100k", "train", "raw", 0.3),
    "ultrachat":   ("openbmb/ultrachat", "train[:5%]", "chat", 0.2),
    "dailydialog": ("roskoN/dailydialog", "train", "chat", 0.1),
}
```

### 6.2 你建议时必须兼容的目录命名
训练配置预期目录：
- `./data/tinystories_rwkv_processed`
- `./data/dailydialog_rwkv_processed`
- `./data/ultrachat_rwkv_processed`
- `./data/minipile_rwkv_processed`

如果你推荐新数据集，请给出建议命名（例如 `./data/fineweb_rwkv_processed`）。

---

## 7. 你可以用的核验命令（给你参考）

### 7.1 检查处理后数据可读
```bash
python scripts/inspect_data.py --path ./data/tinystories_rwkv_processed --n 3
```

### 7.2 检查 tokenizer 一致性
```bash
python scripts/check_vocab.py
```

### 7.3 训练端会读取 mix（你需保证概率和=1）
```python
# train.py 中逻辑会把 probs 归一化，但你仍应给出规范配比
```

---

## 8. 决策原则（你在结论里必须体现）
1. 先保证通用语言建模稳定（base corpus）
2. 再叠加对话/指令数据提升 assistant 行为
3. 再补叙事/高质量小数据调风格
4. 不追求“数据越多越好”，要追求“信号密度 + 许可证安全 + 可持续迭代”

---

## 9. 交付示例（你可仿照）
```markdown
最终建议（示例）:
- fineweb_sample: 0.45
- pile_uncopyrighted: 0.20
- ultrachat: 0.15
- tinystories: 0.10
- cosmopedia: 0.10

原因:
- 用 fineweb + pile 提供广覆盖基础知识
- 用 ultrachat 注入对话式响应
- 用 tiny/cosmo 提升叙事与流畅度
- 整体许可证风险低于直接混用来源不明语料
```

---

## 10. 最终提醒
你是“数据决策顾问”，不是“训练代码工程师”。
你的成功标准是：
- 用户看完就能决定用哪些数据
- 配比能直接进入 `config.py`
- 风险和许可证问题提前说明，不埋雷
