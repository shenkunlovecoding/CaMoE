"""
CaMoE v18 主系统 (Final Fix)
Changes:
1. 强制全程开启 Router (use_market=True)，拒绝随机路由。
2. 修复 LinearTransformerExpert 初始化。
3. 保持 Rescale Trick 和 GC。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple, List
from torch.utils.checkpoint import checkpoint

from .backbone import RWKV7_TimeMix
from .bridge import UltimateBridge
from .experts import SparseRWKVFFN, LinearTransformerExpert
from .critic import CriticVC
from .market import CapitalManager, SparseRouter

class CaMoE_Block(nn.Module):
    """
    单个 CaMoE 块 (支持 Top-2 混合输出)
    """
    
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, 
                 head_size: int, config: Dict, bridge: nn.Module):
        super().__init__()
        
        self.layer_id = layer_id
        self.num_rwkv = config.get('num_rwkv_experts', 6)
        self.num_trans = config.get('num_trans_experts', 2)
        self.num_experts = self.num_rwkv + self.num_trans
        self.n_embd = n_embd
        self.bridge = bridge
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # RWKV-7 TimeMix (Backbone)
        self.att = RWKV7_TimeMix(n_embd, n_layer, layer_id, head_size)
        
        # 专家组
        self.experts = nn.ModuleList()
        
        # RWKV FFN Experts
        for _ in range(self.num_rwkv):
            self.experts.append(SparseRWKVFFN(n_embd))
        
        # Transformer Experts
        n_head = n_embd // head_size
        for _ in range(self.num_trans):
            self.experts.append(LinearTransformerExpert(n_embd, n_head))
        
        # Critic
        self.critic = CriticVC(n_embd, self.num_experts)
    
    def forward(self, 
                x: torch.Tensor, 
                v_first: torch.Tensor,
                capital_shares: torch.Tensor,
                router: SparseRouter,
                step: int,
                warmup_steps: int,
                use_market: bool = True,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        B, T, C = x.shape
        
        # 1. TimeMix (Backbone)
        att_out, v_first, rwkv_state = self.att(self.ln1(x), v_first)
        x = x + att_out
        h = self.ln2(x)
        
        # 2. 计算所有专家的 Confidence
        conf_list = [exp.get_confidence(h) for exp in self.experts]
        confidences = torch.stack(conf_list, dim=-1)  # [B, T, E]
        
        # 3. Market Routing (关键逻辑)
        if not use_market:
            # 只有在极罕见的 Debug 模式下才用随机，训练时严禁进入此分支！
            winners = torch.randint(0, self.num_experts, (B, T, 2), device=x.device)
            weights = torch.ones(B, T, 2, device=x.device) * 0.5
            costs = torch.zeros(B, T, device=x.device)
            difficulty = torch.ones(B, T, 1, device=x.device)
            affinity = torch.zeros(B, T, self.num_experts, device=x.device)
        else:
            difficulty, affinity = self.critic(h)
            critic_subsidy = self.critic.apply_to_bids(torch.zeros_like(confidences), affinity)
            winners, weights, costs, bids = router.route(
                confidences, capital_shares, difficulty, critic_subsidy, training
            )
        
        # 4. 生成 Bridge Prefix (一次性，供所有 Trans 专家使用)
        flat_h = h.reshape(-1, C)
        flat_state = rwkv_state.reshape(-1, C)
        bridge_prefix = self.bridge(flat_h, flat_state)  # [B*T, P, C]
        
        # 5. Top-2 Expert Execution (双路混合)
        final_out = torch.zeros_like(h)  # [B, T, C]
        
        for rank in range(2):
            rank_winners = winners[:, :, rank]  # [B, T]
            rank_weights = weights[:, :, rank].unsqueeze(-1)  # [B, T, 1]
            
            for e in range(self.num_experts):
                mask = (rank_winners == e)  # [B, T]
                if not mask.any():
                    continue
                
                # Gather 被选中的 Token
                selected_h = h[mask]  # [N, C]
                selected_weights = rank_weights[mask]  # [N, 1]
                
                # 执行专家
                if e >= self.num_rwkv:
                    # Transformer: 需要 Prefix
                    flat_mask = mask.reshape(-1)
                    selected_prefix = bridge_prefix[flat_mask]  # [N, P, C]
                    expert_out = self.experts[e](selected_h, selected_prefix)
                else:
                    # RWKV: 不需要 Prefix
                    expert_out = self.experts[e](selected_h, None)
                
                # 加权累加
                weighted_out = expert_out * selected_weights
                final_out[mask] += weighted_out

        # 残差连接
        x = x + final_out
        
        info = {
            "winners": winners,
            "costs": costs,
            "difficulty": difficulty,
            "affinity": affinity,
        }
        return x, v_first, info


class CaMoE_System(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.vocab_size = config['vocab_size']
        
        self.num_rwkv_experts = config.get('num_rwkv_experts', 6)
        self.num_trans_experts = config.get('num_trans_experts', 2)
        self.num_experts = self.num_rwkv_experts + self.num_trans_experts
        
        # Embedding
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        
        # 共享 Bridge
        self.bridge = UltimateBridge(
            self.n_embd, 
            config.get('prefix_len', 64),
            config.get('low_rank_dim', 64)
        )
        
        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_layer):
            self.blocks.append(CaMoE_Block(
                self.n_embd,
                self.n_layer,
                i,
                config['head_size'],
                config,
                bridge=self.bridge
            ))
        
        self.ln_out = nn.LayerNorm(self.n_embd)
        
        # Head (可选 Tied Embedding)
        if config.get('tied_embeddings', False):
            self.head = None  # 使用 emb.weight
        else:
            self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Market
        self.capital_manager = CapitalManager(
            self.n_layer, self.num_experts,
            total_capital=config.get('total_capital', 10000.0),
            min_share=config.get('min_capital_share', 0.05),
            tax_threshold=config.get('tax_threshold', 2.0),
            tax_rate=config.get('tax_rate', 0.1)
        )
        
        self.router = SparseRouter()
    
    def forward(self, idx: torch.Tensor, step: int = 0, 
                phase: str = "normal") -> Tuple[torch.Tensor, Dict]:
        x = self.emb(idx)
        v_first = None
        
        # [CRITICAL FIX] 始终开启 Market Routing
        # 即使在 Prewarm/Warmup，我们也需要 Router 选出最好的专家，让专家获得正确的梯度
        # 资本的更新 (Update) 由 train.py 控制，这里只管路由 (Selection)
        use_market = True 
        
        all_info = {
            "winners": [], "costs": [], "difficulties": [], "affinities": []
        }
        warmup_steps = self.config.get('warmup_steps', 2000)

        for i, block in enumerate(self.blocks):
            shares = self.capital_manager.get_shares(i)
            if self.training:
                # 开启 Checkpoint 以节省显存
                x, v_first, info = checkpoint(
                    block, 
                    x, v_first, shares, self.router, 
                    step, warmup_steps, use_market, True, # training=True
                    use_reentrant=False
                )
            else:
                x, v_first, info = block(
                    x, v_first, shares, self.router,
                    step, warmup_steps, use_market, self.training
                )
            
            all_info["winners"].append(info["winners"].detach())
            all_info["costs"].append(info["costs"].detach())
            all_info["difficulties"].append(info["difficulty"].detach())
            all_info["affinities"].append(info["affinity"].detach())
        
        x = self.ln_out(x)
        
        # Output (Tied Embedding Rescale Trick)
        if self.head is not None:
            logits = self.head(x)
        else:
            #x = x * (self.n_embd ** -0.5) 
            logits = F.linear(x, self.emb.weight)
        
        return logits, all_info
    
    def compute_losses(self, logits, targets, all_info):
        B, T = targets.shape
        
        # Main Loss
        main_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        
        # Token Losses (for Market Update)
        with torch.no_grad():
            token_losses = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction='none'
            ).reshape(B, T)
        
        # Critic Loss
        critic_loss = 0.0
        for i, diff in enumerate(all_info.get("difficulties", [])):
            baseline = self.capital_manager.baseline_losses[i]
            target = F.relu(token_losses - baseline)
            critic_loss += F.smooth_l1_loss(diff.squeeze(-1), target)
        
        if len(all_info.get("difficulties", [])) > 0:
            critic_loss /= len(all_info["difficulties"])

        total_loss = main_loss + 0.1 * critic_loss
        bridge_loss = 0.0 # No longer used
        
        return total_loss, token_losses, main_loss, critic_loss, bridge_loss
    
    def update_market(self, all_info, token_losses, step):
        with torch.no_grad():
            for i in range(self.n_layer):
                if i >= len(all_info.get("winners", [])): 
                    continue
                
                self.capital_manager.update(
                    i, all_info["winners"][i], token_losses, all_info["costs"][i]
                )
                
                baseline = self.capital_manager.baseline_losses[i].item()
                self.blocks[i].critic.settle(
                    all_info["affinities"][i], all_info["winners"][i],
                    token_losses, baseline
                )

                # Bailout logic
                if self.blocks[i].critic.capital < 200:
                    self.blocks[i].critic.capital.fill_(2000.0)
                    if step % 100 == 0:
                        print(f"🏛️  Layer {i}: Critic Bailout (Step {step})")
    
    def log_market_health(self) -> Dict:
        metrics = {}
        check_layers = [0, self.n_layer // 2, self.n_layer - 1]
        check_layers = sorted(list(set([i for i in check_layers if i < self.n_layer])))

        for i in check_layers:
            caps = self.capital_manager.capitals[i]
            total_cap = caps.sum() + 1e-6
            
            rwkv_share = caps[:self.blocks[i].num_rwkv].sum() / total_cap * 100
            trans_share = caps[self.blocks[i].num_rwkv:].sum() / total_cap * 100
            
            sorted_caps, _ = torch.sort(caps)
            n = self.num_experts
            idx = torch.arange(1, n + 1, device=caps.device, dtype=caps.dtype)
            gini = ((2 * idx - n - 1) * sorted_caps).sum() / (n * total_cap + 1e-6)
            
            metrics[f"L{i}/TransShare"] = trans_share.item()
            metrics[f"L{i}/RWKVShare"] = rwkv_share.item()
            metrics[f"L{i}/Gini"] = gini.item()
            metrics[f"L{i}/CriticCap"] = self.blocks[i].critic.capital.item()
        
        return metrics