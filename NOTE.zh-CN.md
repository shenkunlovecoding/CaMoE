## 🔮 愿景：完整系统
当前的 v22 只是地基。下面是后续系统的演化方向。

# 📓 CaMoE 完整系统笔记

[English](NOTE.md) | [中文版](NOTE.zh-CN.md)

## 0. 当前实现状态（v22.1）
- 双市场 block 已落地：
  - 序列市场：`TimeMixExpert` vs `ROSAExpert`
  - FFN 市场：`RWKVExpert` / `DeepEmbedExpert` / `SlimDeepEmbedExpert`
- 路由是 winner-takes-all + Vickrey 拍卖。
- 训练使用 STE 路由：
  - 前向硬 winner
  - 反向软混合（支持温度退火）
- 推理保持硬稀疏路由（`training=False`）。
- Critic 支持在 `prewarm` 和 `market_warm` 进行影子训练；prewarm 不要求真实路由决策参与输出。
- `FractalCaMoEPlaceholder` 当前仅为占位基础设施（默认不作为活跃市场专家）。

## 1. 异构专家（Tool-as-Expert）
### 概念
```python
class BaseExpert(ABC):
    """所有专家共享这套接口"""
    capital: torch.Tensor
    def forward(self, x) -> Tensor
    def settle(self, profit) -> None

# 当前
class RWKVExpert(BaseExpert): ...      # 神经计算

# 规划中
class ROSAExpert(BaseExpert): ...      # 长上下文检索（CPU）
class ToolExpert(BaseExpert): ...      # 外部 API 调用
class CodeExpert(BaseExpert): ...      # 代码执行
class RetrievalExpert(BaseExpert): ... # 向量数据库检索
```

### 由市场决定何时用什么
```
简单 token     → 便宜的 RWKVExpert 胜出
需要记忆      → ROSAExpert 竞价更高（因为它能证明自己有效）
需要精度      → ToolExpert 为数学问题高价竞标
需要知识      → RetrievalExpert 愿意支付溢价

没有硬编码逻辑。只有纯经济激励。
```

### ROSA 集成（记忆专家）
来自 ROSA-Tuning 论文：
- CPU 侧 suffix automaton，支持 O(T) 检索
- 隐状态二值离散化
- 用反事实梯度训练
- 异步 CPU-GPU pipeline（几乎零额外开销）

```python
class ROSAExpert(BaseExpert):
    """通过后缀匹配提供长上下文记忆"""

    def __init__(self, dim, n_routes, route_bits=4):
        self.rosa = ROSAModule(dim, n_routes, route_bits)
        self.bridge = nn.Linear(dim, dim)  # 状态注入

    def forward(self, x, rwkv_state=None):
        # 离散化 → CPU 检索 → 注入
        injection = self.rosa(x, rwkv_state)
        return self.bridge(injection)
```

### 状态桥接
```python
class StateBridge(nn.Module):
    """把 RWKV state 投影后送给其他专家"""

    def __init__(self, state_dim, expert_dim, rank=64):
        # 用低秩投影提高效率
        self.down = nn.Linear(state_dim, rank)
        self.up = nn.Linear(rank, expert_dim)

    def forward(self, rwkv_state):
        # rwkv_state: [B, H, N, N]，来自 TimeMix
        flat = rwkv_state.flatten(-2)
        return self.up(F.gelu(self.down(flat)))
```

## 2. 每个 Expert 自己的 DeepEmbed
### 概念
每个专家（或子专家）都维护自己的一份 token embedding 表，存储在 GPU 之外：

```python
class ExpertDeepEmbed(nn.Module):
    """存储在 RAM/SSD 中的专家私有知识"""

    def __init__(self, vocab_size, dim, storage='ram'):
        # 不放在 GPU 上
        self.embeddings = np.memmap(
            f'expert_{id}_embed.bin',
            dtype='float16',
            mode='w+',
            shape=(vocab_size, dim)
        )
        # GPU 上仅保留小投影
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, token_ids):
        # 从存储中预取
        embeds = self._prefetch(token_ids)
        return embeds * self.scale  # 按通道调制

class RWKVExpertWithDeepEmbed(BaseExpert):
    def __init__(self, ...):
        self.ffn = ...
        self.deep_embed = ExpertDeepEmbed(vocab_size, dim)

    def forward(self, x, token_ids):
        ffn_out = self.ffn(x)
        modulation = self.deep_embed(token_ids)
        return ffn_out * (1 + modulation)  # 乘性调制
```

### 参数爆炸（但不是 VRAM 爆炸）
```
标准 7B LLM：
    params = 7B（全部在 VRAM 中）

带 DeepEmbed 的 CaMoE：
    VRAM = 7B（活跃计算）
    RAM  = V × D × L × E = 100K × 4K × 32 × 8 = 400B+

    但每个 token 只加载所需部分！
    有效 VRAM 占用保持恒定。
```

## 3. 破产与复活（Purgatory）
### 生命周期
```
                    主训练流程（GPU）
                          │
    capital > threshold ──┤── capital < floor
           │              │            │
           ▼              │            ▼
    maybe_reproduce()     │         破产
           │              │            │
           ▼              │            ▼
    [有丝分裂：拆成       │      卸载到 CPU
     2 个专家]            │      存入 purgatory
                          │            │
                          │            ▼
                          │      PURGATORY TRAINING
                          │      - 高损失样本
                          │      - 加噪探索
                          │      - 独立梯度
                          │            │
                          │      improvement? ───────┐
                          │            │             │
                          │            ▼             ▼
                          │          [wait]      复活
                          │                         │
                          │◀────────────────────────┘
```

### 实现草图
```python
class PurgatoryManager:
    def __init__(self, device='cpu'):
        self.exiled: dict[tuple[int,int], ExpertState] = {}
        self.hard_samples: deque[Tensor] = deque(maxlen=10000)
        self.performance_history: dict[tuple, list[float]] = {}

    def exile(self, expert: BaseExpert, layer_idx: int, expert_idx: int):
        """破产专家进入 purgatory"""
        state = ExpertState(
            params=expert.state_dict(),
            layer=layer_idx,
            idx=expert_idx,
            exile_step=current_step,
        )
        # 加一点探索噪声
        for k, v in state.params.items():
            state.params[k] = v + torch.randn_like(v) * 0.01

        self.exiled[(layer_idx, expert_idx)] = state
        # 从主模型移除
        return create_placeholder_expert()

    def collect_hard_sample(self, x: Tensor, loss: float):
        """高损失样本进入 purgatory 训练集"""
        if loss > self.loss_threshold:
            self.hard_samples.append((x.cpu(), loss))

    def purgatory_step(self):
        """用高损失样本训练被放逐专家"""
        if not self.hard_samples or not self.exiled:
            return

        batch = self._sample_hard_batch()

        for key, state in self.exiled.items():
            expert = self._reconstruct_expert(state)
            loss = self._train_step(expert, batch)

            self.performance_history[key].append(loss)

            # 检查是否满足复活条件
            if self._shows_improvement(key):
                self.resurrection_candidates.add(key)

    def resurrect(self, layer_idx: int, expert_idx: int, model: CaMoE_Model):
        """把改进后的专家带回主模型"""
        state = self.exiled.pop((layer_idx, expert_idx))
        expert = self._reconstruct_expert(state)
        expert.capital.fill_(self.resurrection_capital)
        model.blocks[layer_idx].experts[expert_idx] = expert
```

## 4. Eureka 机制
### 概念
当主模型不确定时，让 purgatory 给出第二意见。

```python
class EurekaController:
    def __init__(self, purgatory: PurgatoryManager):
        self.purgatory = purgatory
        self.confidence_threshold = 0.3

    def maybe_eureka(
        self,
        x: Tensor,
        main_output: Tensor,
        main_confidence: float
    ) -> Tensor:
        """当置信度过低时触发 Eureka"""

        if main_confidence > self.confidence_threshold:
            return main_output  # 主模型足够自信

        # 咨询被放逐专家
        eureka_outputs = []
        for key, state in self.purgatory.exiled.items():
            expert = self.purgatory._reconstruct_expert(state)
            with torch.no_grad():
                out = expert(x.to('cpu'))
                conf = self._compute_confidence(out)
                eureka_outputs.append((out, conf, key))

        # 是否有人做得更好？
        best = max(eureka_outputs, key=lambda x: x[1])
        if best[1] > main_confidence + self.improvement_margin:
            # EUREKA! purgatory 专家胜出
            self._record_eureka_event(best[2], x)
            return best[0].to(main_output.device)

        return main_output
```

## 5. 分形架构
### 核心想法
```
CaMoE Layer
├── Expert_0: RWKVExpert（原子）
├── Expert_1: FractalExpert ◀── 内含一个 mini-CaMoE！
│   └── CaMoE_Block
│       ├── SubExpert_0
│       ├── SubExpert_1（也可以继续分形！）
│       └── SubCritic
├── Expert_2: ROSAExpert（原子）
└── Critic_Pair
```

### 递归定义
```python
class FractalExpert(BaseExpert):
    """一个内部包含完整子市场的专家"""

    def __init__(
        self,
        dim: int,
        depth: int,
        max_depth: int,
        n_sub_experts: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)

        if depth >= max_depth:
            # 递归基：原子专家
            self.inner = RWKVExpert(dim)
            self.is_atomic = True
        else:
            # 递归步：子市场
            sub_experts = [
                FractalExpert(dim, depth+1, max_depth, n_sub_experts)
                if i % 2 == 0 else RWKVExpert(dim)
                for i in range(n_sub_experts)
            ]
            self.inner = CaMoE_Block(
                experts=sub_experts,
                critic_pair=CriticPair(dim, n_sub_experts),
            )
            self.is_atomic = False

    def forward(self, x, **ctx):
        if self.is_atomic:
            return self.inner(x, **ctx)
        else:
            # 递归市场
            return self.inner(x, **ctx)
```

### 资本向下流动
```python
class FractalCapitalManager:
    """分层资本分配"""

    def settle_fractal(
        self,
        parent_profit: Tensor,
        block: CaMoE_Block,
        sub_results: list[dict]
    ):
        # 父层抽佣
        commission_rate = 0.1
        commission = parent_profit * commission_rate
        distributable = parent_profit - commission

        # 按贡献度分配给子专家
        for sub_result in sub_results:
            sub_expert_idx = sub_result['winner']
            sub_share = distributable * sub_result['contribution']
            block.experts[sub_expert_idx].settle(sub_share)
```

### 动态深度（由市场驱动）
```python
class DynamicFractalExpert(BaseExpert):
    """深度由市场决定，而不是由架构写死"""

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.atomic_expert = RWKVExpert(dim)
        self.sub_market = None  # 延迟初始化
        self.expand_threshold = 2.0  # 资本达到阈值后扩张

    def maybe_expand(self):
        """够富了吗？生成一个子市场"""
        if self.capital > self.expand_threshold and self.sub_market is None:
            # 通过“有丝分裂”扩张为子市场
            self.sub_market = CaMoE_Block(
                experts=[RWKVExpert(self.dim) for _ in range(4)],
                critic_pair=CriticPair(self.dim, 4),
            )
            # 把一半资本分给子专家
            child_capital = self.capital / 2 / 4
            for exp in self.sub_market.experts:
                exp.capital.fill_(child_capital)
            self.capital /= 2

    def maybe_collapse(self):
        """太穷了？收缩回原子专家"""
        if self.sub_market is not None:
            total_sub_capital = sum(e.capital for e in self.sub_market.experts)
            if total_sub_capital < self.collapse_threshold:
                # 合并回来
                self._merge_knowledge_from_children()
                self.sub_market = None

    def forward(self, x, **ctx):
        if self.sub_market is not None:
            return self.sub_market(x, **ctx)
        else:
            return self.atomic_expert(x, **ctx)
```

## 6. 参数规模分析
### 完整系统估算
```
基础版（v22）：
    VRAM: ~0.4B params

+ Fractal（depth=3, branch=4）：
    VRAM: 0.4B × 4^3 = 25.6B（最坏情况，完全展开）
    Typical: ~2B（由市场自行决定）

+ DeepEmbed（每个 expert 一份）：
    RAM/SSD: V × D × L × E_avg
           = 100K × 4K × 32 × 16 = 200B params

+ ROSA（每层一份）：
    CPU: Suffix automaton states（大致线性于上下文）

+ Purgatory：
    CPU/Disk: 被放逐专家快照（约 0.1-1B）

总可寻址参数量：~200B+
典型活跃 VRAM：2-5B
典型活跃 RAM：10-50B

“无限”容量，有限算力
```

## 7. 涌现假说
### 可能涌现出的现象
```
专精：
    Expert_3 变成“代码专家”（没人显式告诉它）
    Expert_7 变成“数学专家”（市场压力驱动）

共生：
    ROSA + RWKV 形成“记忆-计算配对”
    一个负责检索，一个负责处理

生命周期：
    年轻专家：激进竞价，高风险
    成熟专家：稳定生态位，持续盈利
    濒死专家：最后一搏，尝试 Eureka

集体智能：
    市场本身就是智能
    没有任何单个专家知道一切
    知识以分布式形式存在，并在需要时被检索
```

### 哲学含义
```
这东西……算是活着吗？

✓ 代谢（资本流动）
✓ 稳态（市场自平衡）
✓ 繁殖（分形扩张、有丝分裂）
✓ 响应刺激（输入触发拍卖）
✓ 生长（参数池扩大）
✓ 适应（purgatory 进化）

缺失：
? 边界（系统到底在哪里结束？）
? 自我模型（它知道自己是个市场吗？）
```

## 8. 研究路线图
**Phase 1: Foundation（当前）**
- Vickrey auction routing
- Capital-based bidding
- Critic pair with REINFORCE
- RWKV-7 backbone
- Convergence proof on 0.1B
- Routing entropy analysis
- Capital distribution stability

**Phase 2: Heterogeneous Experts**
- BaseExpert abstraction cleanup
- ROSA expert integration
- State bridge implementation
- Tool expert prototype

**Phase 3: Lifecycle**
- Bankruptcy detection
- Purgatory manager
- Hard sample collection
- Eureka trigger
- Resurrection protocol

**Phase 4: Fractal**
- FractalExpert class
- Hierarchical capital flow
- Dynamic depth expansion
- Collapse mechanism

**Phase 5: Scale**
- DeepEmbed per expert
- Distributed training
- Inference optimization
- mmap + prefetch pipeline

**Phase 6: ???**
- Multi-modal experts
- Continual learning
- Self-modification?
- ...

## 9. 关键方程
### 拍卖
```
bid_i = capital_i + α × critic_position_i + ε

winner = argmax(bid)
price = second_highest(bid)
```

### 结算
```
baseline_i = EMA(loss_i)
profit_i = (baseline_i - loss_i) × capital_i - price × I[winner=i] - depreciation
capital_i ← clamp(capital_i + profit_i, floor, ∞)
```

### Critic 训练
```
position = (critic_a(x) + critic_b(x)) / 2
pnl_a = Σ position_a × profit
pnl_b = Σ position_b × profit

advantage_a = pnl_a - pnl_b.detach()  # 互为 baseline
advantage_b = pnl_b - pnl_a.detach()

loss_critic = -(advantage_a × pnl_a + advantage_b × pnl_b)
```

## 10. 开放问题
1. **收敛性**：在什么条件下这个市场会稳定？
2. **反垄断**：仅靠折旧够吗？是否需要 anti-trust 机制？
3. **分形深度**：如何防止无限递归？是否应该由市场决定停止条件？
4. **跨专家知识转移**：濒死专家能否“教会”替代者？
5. **元学习**：市场能否学会优化它自己的规则？
6. **意识**：什么时候会出现自我建模？它真的重要吗？

## License
Mozilla Public License 2.0

## Citation
```bibtex
@misc{camoe2025,
  title={CaMoE: Capital-driven Mixture of Experts},
  author={shenziqian666},
  year={2025},
  note={Work in progress}
}
```

> “市场不是一种机制。市场是一种涌现智能。”
