## 🔮 Vision: The Complete System
The current v22 is the foundation. Below is where we're heading.

# 📓 CaMoE Complete System Notes

[中文版](NOTE.zh-CN.md) | [English](NOTE.md)

## 1. Heterogeneous Experts (Tool-as-Expert)
### Concept
```python
class BaseExpert(ABC):
    """All experts share this interface"""
    capital: torch.Tensor
    def forward(self, x) -> Tensor
    def settle(self, profit) -> None

# Current
class RWKVExpert(BaseExpert): ...      # Neural compute

# Planned
class ROSAExpert(BaseExpert): ...      # Long-context retrieval (CPU)
class ToolExpert(BaseExpert): ...      # External API calls
class CodeExpert(BaseExpert): ...      # Code execution
class RetrievalExpert(BaseExpert): ... # Vector DB lookup
```

### Market Decides When to Use What
```
Simple token     → cheap RWKVExpert wins
Need memory      → ROSAExpert outbids (has proof it helps)
Need precision   → ToolExpert bids high for math
Need knowledge   → RetrievalExpert pays premium

No hardcoded logic. Pure economic incentives.
```

### ROSA Integration (Memory Expert)
From ROSA-Tuning paper:
- CPU-side suffix automaton for O(T) retrieval
- Binary discretization of hidden states
- Counterfactual gradients for training
- Async CPU-GPU pipeline (nearly zero overhead)

```python
class ROSAExpert(BaseExpert):
    """Long-context memory via suffix matching"""
    
    def __init__(self, dim, n_routes, route_bits=4):
        self.rosa = ROSAModule(dim, n_routes, route_bits)
        self.bridge = nn.Linear(dim, dim)  # State injection
    
    def forward(self, x, rwkv_state=None):
        # Discretize → CPU retrieval → inject
        injection = self.rosa(x, rwkv_state)
        return self.bridge(injection)
```

### State Bridge
```python
class StateBridge(nn.Module):
    """Project RWKV state to feed other experts"""
    
    def __init__(self, state_dim, expert_dim, rank=64):
        # Low-rank projection for efficiency
        self.down = nn.Linear(state_dim, rank)
        self.up = nn.Linear(rank, expert_dim)
    
    def forward(self, rwkv_state):
        # rwkv_state: [B, H, N, N] from TimeMix
        flat = rwkv_state.flatten(-2)
        return self.up(F.gelu(self.down(flat)))
```

## 2. DeepEmbed Per Expert
### Concept
Each expert (or sub-expert) maintains its own token embedding table stored off-GPU:

```python
class ExpertDeepEmbed(nn.Module):
    """Per-expert knowledge stored in RAM/SSD"""
    
    def __init__(self, vocab_size, dim, storage='ram'):
        # NOT on GPU
        self.embeddings = np.memmap(
            f'expert_{id}_embed.bin',
            dtype='float16',
            mode='w+',
            shape=(vocab_size, dim)
        )
        # Small projection on GPU
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, token_ids):
        # Prefetch from storage
        embeds = self._prefetch(token_ids)
        return embeds * self.scale  # Channel-wise modulation

class RWKVExpertWithDeepEmbed(BaseExpert):
    def __init__(self, ...):
        self.ffn = ...
        self.deep_embed = ExpertDeepEmbed(vocab_size, dim)
    
    def forward(self, x, token_ids):
        ffn_out = self.ffn(x)
        modulation = self.deep_embed(token_ids)
        return ffn_out * (1 + modulation)  # Multiplicative
```

### Parameter Explosion (But Not VRAM)
```
Standard 7B LLM:
    params = 7B (all in VRAM)

CaMoE with DeepEmbed:
    VRAM = 7B (active compute)
    RAM  = V × D × L × E = 100K × 4K × 32 × 8 = 400B+
    
    But only load what's needed per token!
    Effective VRAM usage stays constant.
```

## 3. Bankruptcy & Resurrection (Purgatory)
### The Lifecycle
```
                    MAIN TRAINING (GPU)
                          │
    capital > threshold ──┤── capital < floor
           │              │            │
           ▼              │            ▼
    maybe_reproduce()     │     BANKRUPTCY
           │              │            │
           ▼              │            ▼
    [mitosis: split       │     offload to CPU
     into 2 experts]      │     store in purgatory
                          │            │
                          │            ▼
                          │     PURGATORY TRAINING
                          │     - high loss samples
                          │     - add noise exploration
                          │     - independent gradient
                          │            │
                          │     improvement? ───────┐
                          │            │            │
                          │            ▼            ▼
                          │         [wait]    RESURRECTION
                          │                        │
                          │◀───────────────────────┘
```

### Implementation Sketch
```python
class PurgatoryManager:
    def __init__(self, device='cpu'):
        self.exiled: dict[tuple[int,int], ExpertState] = {}
        self.hard_samples: deque[Tensor] = deque(maxlen=10000)
        self.performance_history: dict[tuple, list[float]] = {}
    
    def exile(self, expert: BaseExpert, layer_idx: int, expert_idx: int):
        """Bankrupt expert goes to purgatory"""
        state = ExpertState(
            params=expert.state_dict(),
            layer=layer_idx,
            idx=expert_idx,
            exile_step=current_step,
        )
        # Add exploration noise
        for k, v in state.params.items():
            state.params[k] = v + torch.randn_like(v) * 0.01
        
        self.exiled[(layer_idx, expert_idx)] = state
        # Remove from main model
        return create_placeholder_expert()
    
    def collect_hard_sample(self, x: Tensor, loss: float):
        """High-loss samples go to purgatory training set"""
        if loss > self.loss_threshold:
            self.hard_samples.append((x.cpu(), loss))
    
    def purgatory_step(self):
        """Train exiled experts on hard samples"""
        if not self.hard_samples or not self.exiled:
            return
        
        batch = self._sample_hard_batch()
        
        for key, state in self.exiled.items():
            expert = self._reconstruct_expert(state)
            loss = self._train_step(expert, batch)
            
            self.performance_history[key].append(loss)
            
            # Check for resurrection
            if self._shows_improvement(key):
                self.resurrection_candidates.add(key)
    
    def resurrect(self, layer_idx: int, expert_idx: int, model: CaMoE_Model):
        """Bring back improved expert"""
        state = self.exiled.pop((layer_idx, expert_idx))
        expert = self._reconstruct_expert(state)
        expert.capital.fill_(self.resurrection_capital)
        model.blocks[layer_idx].experts[expert_idx] = expert
```

## 4. Eureka Mechanism
### Concept
When main model is uncertain → ask purgatory for second opinion

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
        """Trigger Eureka when confidence is low"""
        
        if main_confidence > self.confidence_threshold:
            return main_output  # Main model is confident
        
        # Ask exiled experts
        eureka_outputs = []
        for key, state in self.purgatory.exiled.items():
            expert = self.purgatory._reconstruct_expert(state)
            with torch.no_grad():
                out = expert(x.to('cpu'))
                conf = self._compute_confidence(out)
                eureka_outputs.append((out, conf, key))
        
        # Did anyone do better?
        best = max(eureka_outputs, key=lambda x: x[1])
        if best[1] > main_confidence + self.improvement_margin:
            # EUREKA! Purgatory expert wins
            self._record_eureka_event(best[2], x)
            return best[0].to(main_output.device)
        
        return main_output
```

## 5. Fractal Architecture
### The Big Idea
```
CaMoE Layer
├── Expert_0: RWKVExpert (atomic)
├── Expert_1: FractalExpert ◀── contains mini-CaMoE!
│   └── CaMoE_Block
│       ├── SubExpert_0
│       ├── SubExpert_1 (could be fractal too!)
│       └── SubCritic
├── Expert_2: ROSAExpert (atomic)
└── Critic_Pair
```

### Recursive Definition
```python
class FractalExpert(BaseExpert):
    """An expert that contains an entire sub-market"""
    
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
            # Base case: atomic experts
            self.inner = RWKVExpert(dim)
            self.is_atomic = True
        else:
            # Recursive case: sub-market
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
            # Recursive market!
            return self.inner(x, **ctx)
```

### Capital Flows Down
```python
class FractalCapitalManager:
    """Hierarchical capital distribution"""
    
    def settle_fractal(
        self,
        parent_profit: Tensor,
        block: CaMoE_Block,
        sub_results: list[dict]
    ):
        # Parent takes commission
        commission_rate = 0.1
        commission = parent_profit * commission_rate
        distributable = parent_profit - commission
        
        # Distribute to sub-experts proportional to their contribution
        for sub_result in sub_results:
            sub_expert_idx = sub_result['winner']
            sub_share = distributable * sub_result['contribution']
            block.experts[sub_expert_idx].settle(sub_share)
```

### Dynamic Depth (Market-Driven)
```python
class DynamicFractalExpert(BaseExpert):
    """Depth decided by market, not architecture"""
    
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.atomic_expert = RWKVExpert(dim)
        self.sub_market = None  # Lazy init
        self.expand_threshold = 2.0  # Capital threshold to expand
    
    def maybe_expand(self):
        """Rich enough? Spawn a sub-market"""
        if self.capital > self.expand_threshold and self.sub_market is None:
            # Mitosis into sub-market
            self.sub_market = CaMoE_Block(
                experts=[RWKVExpert(self.dim) for _ in range(4)],
                critic_pair=CriticPair(self.dim, 4),
            )
            # Transfer half capital to children
            child_capital = self.capital / 2 / 4
            for exp in self.sub_market.experts:
                exp.capital.fill_(child_capital)
            self.capital /= 2
    
    def maybe_collapse(self):
        """Too poor? Collapse back to atomic"""
        if self.sub_market is not None:
            total_sub_capital = sum(e.capital for e in self.sub_market.experts)
            if total_sub_capital < self.collapse_threshold:
                # Merge back
                self._merge_knowledge_from_children()
                self.sub_market = None
    
    def forward(self, x, **ctx):
        if self.sub_market is not None:
            return self.sub_market(x, **ctx)
        else:
            return self.atomic_expert(x, **ctx)
```

## 6. Parameter Scale Analysis
### Full System Estimation
```
Base (v22):
    VRAM: ~0.4B params
    
+ Fractal (depth=3, branch=4):
    VRAM: 0.4B × 4^3 = 25.6B (worst case, full expansion)
    Typical: ~2B (market decides)

+ DeepEmbed (per expert):
    RAM/SSD: V × D × L × E_avg
           = 100K × 4K × 32 × 16 = 200B params
           
+ ROSA (per layer):
    CPU: Suffix automaton states (~linear in context)
    
+ Purgatory:
    CPU/Disk: Exiled expert snapshots (~0.1-1B)

Total Addressable Parameters: ~200B+
Typical Active VRAM: 2-5B
Typical Active RAM: 10-50B

"Infinite" capacity, finite compute
```

## 7. The Emergence Hypothesis
### What Might Emerge
```
Specialization:
    Expert_3 becomes "the code expert" (no one told it to)
    Expert_7 becomes "the math expert" (market pressure)
    
Symbiosis:
    ROSA + RWKV form "memory-compute pairs"
    One retrieves, one processes
    
Lifecycle:
    Young experts: aggressive bidding, high risk
    Mature experts: stable niche, consistent profit
    Dying experts: last-ditch Eureka attempts
    
Collective Intelligence:
    The market IS the intelligence
    No single expert knows everything
    Knowledge is distributed + retrieved on demand
```

### Philosophical Implications
```
Is this... alive?

✓ Metabolism (capital flow)
✓ Homeostasis (market self-balances)
✓ Reproduction (fractal expansion, mitosis)
✓ Response to stimuli (auction on input)
✓ Growth (parameter pool grows)
✓ Adaptation (purgatory evolution)

Missing:
? Boundary (where does the system end?)
? Self-model (does it know it's a market?)
```

## 8. Research Roadmap
**Phase 1: Foundation (Current)**
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

## 9. Key Equations
### Auction
```
bid_i = capital_i + α × critic_position_i + ε

winner = argmax(bid)
price = second_highest(bid)
```

### Settlement
```
baseline_i = EMA(loss_i)
profit_i = (baseline_i - loss_i) × capital_i - price × I[winner=i] - depreciation
capital_i ← clamp(capital_i + profit_i, floor, ∞)
```

### Critic Training
```
position = (critic_a(x) + critic_b(x)) / 2
pnl_a = Σ position_a × profit
pnl_b = Σ position_b × profit

advantage_a = pnl_a - pnl_b.detach()  # Mutual baseline
advantage_b = pnl_b - pnl_a.detach()

loss_critic = -(advantage_a × pnl_a + advantage_b × pnl_b)
```

## 10. Open Questions
1. **Convergence**: Under what conditions does the market stabilize?
2. **Monopoly Prevention**: Is depreciation enough? Need anti-trust?
3. **Fractal Depth**: How to prevent infinite recursion? Market-based stopping?
4. **Cross-Expert Knowledge**: Can dying experts "teach" their replacements?
5. **Meta-Learning**: Can the market learn to improve its own rules?
6. **Consciousness**: At what scale does self-modeling emerge? Does it matter?

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

> "The market is not a mechanism. The market is an emergent intelligence."
