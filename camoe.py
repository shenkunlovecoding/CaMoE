"""
CaMoE v10.0 主系统
稀疏激活版 - 只让胜者计算FFN
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple, List
import torch.utils.checkpoint as checkpoint
from backbone import RWKV7_TimeMix
from bridge import UltimateBridge
from experts import SparseRWKVFFN, LinearTransformerExpert
from critic import CriticVC
from market import CapitalManager, SparseRouter, EurekaController


class CaMoE_Block(nn.Module):
    """
    单个CaMoE块 (稀疏激活)
    """
    
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, 
                 head_size: int, num_rwkv_experts: int, n_state):
        super().__init__()
        
        self.layer_id = layer_id
        self.num_experts = num_rwkv_experts + 1  # +1 for Trans
        self.n_embd = n_embd
        self.bridge = UltimateBridge(n_embd, max_prefix_len=64)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # RWKV-7 TimeMix
        self.att = RWKV7_TimeMix(n_embd, n_layer, layer_id, head_size)
        
        # 专家组
        self.experts = nn.ModuleList()
        for _ in range(num_rwkv_experts):
            self.experts.append(SparseRWKVFFN(n_embd))
        self.experts.append(LinearTransformerExpert(n_embd, n_head=n_embd//head_size, bridge=self.bridge))
        
        # Critic
        self.critic = CriticVC(n_embd, self.num_experts)
    
    def forward(self, 
                x: torch.Tensor, 
                v_first: torch.Tensor,
                capital_shares: torch.Tensor,
                router: SparseRouter,
                use_market: bool = True,
                eureka_override: torch.Tensor = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        B, T, C = x.shape
        
        # TimeMix
        att_out, v_first,rwkv_state = self.att(self.ln1(x), v_first)
        x = x + att_out
        h = self.ln2(x)
        
        if not use_market:
            # Warmup: 随机路由
            winners = torch.randint(0, self.num_experts, (B, T), device=x.device)
            costs = torch.zeros(B, T, device=x.device)
            difficulty = torch.ones(B, T, 1, device=x.device)
            affinity = torch.zeros(B, T, self.num_experts, device=x.device)
            confidences = torch.ones(B, T, self.num_experts, device=x.device)
        else:
            # ===== 第一步: 所有专家报价 =====
            # [修改点] List 改名，防止混淆
            conf_list = [] 
            for exp in self.experts:
                # 传入 rwkv_state (如果专家需要的话，FFN 不需要但 Trans 需要，这里 h 是输入)
                # 注意：Confidence 只基于 h 计算
                conf = exp.get_confidence(h) 
                conf_list.append(conf)
            
            # [修改点] Stack 后的 Tensor 命名为 confidences
            confidences = torch.stack(conf_list, dim=-1)  # [B, T, E]
            
            # Critic预测
            difficulty, affinity = self.critic(h)
            
            # 路由决策
            # [修改点] 这里用 confidences (Tensor)
            critic_subsidy = self.critic.apply_to_bids(
                torch.zeros_like(confidences), affinity
            )
            winners, costs, bids = router.route(
                confidences, capital_shares, difficulty, critic_subsidy, training
            )
            
            # Eureka覆盖
            if eureka_override is not None:
                valid = eureka_override >= 0
                winners = torch.where(valid, eureka_override, winners)
        flat_h = h.reshape(-1, C)              # [N_total, C]
        flat_state = rwkv_state.reshape(-1, C) # [N_total, C]
        flat_winners = winners.reshape(-1)     # [N_total]
        
        # 3. 梯度直通 (STE / Confidence Scaling)
        # 取出每个 token 对应的胜者 confidence
        flat_conf = confidences.view(-1, self.num_experts) # [N_total, E]
        row_idx = torch.arange(flat_h.shape[0], device=x.device)
        winning_bids = flat_conf[row_idx, flat_winners].unsqueeze(-1) # [N_total, 1]
        
        # Scaling Factor: 数值为 1，但携带梯度
        scale_factor = winning_bids / (winning_bids.detach() + 1e-6)
        
        # 4. 稀疏循环 & Gather/Scatter
        final_out = torch.zeros_like(flat_h)
        total_recon_loss = 0.0
        
        for e in range(self.num_experts):
            # Find tokens for this expert
            mask_indices = (flat_winners == e).nonzero(as_tuple=True)[0]
            if mask_indices.numel() == 0:
                continue
            
            # Gather
            selected_h = flat_h[mask_indices]
            selected_state = flat_state[mask_indices]
            selected_scale = scale_factor[mask_indices]
            
            # Forward
            if e == self.num_experts - 1: # Transformer
                expert_out, recon_loss = self.experts[e](selected_h, selected_state)
                total_recon_loss += recon_loss
            else: # RWKV FFN
                expert_out, _ = self.experts[e](selected_h, selected_state)
            
            # Apply Gradient Scaling
            expert_out_scaled = expert_out * selected_scale
            expert_out_scaled = expert_out_scaled.to(dtype=final_out.dtype)
            # Scatter
            final_out.index_copy_(0, mask_indices, expert_out_scaled)
            
        out = final_out.reshape(B, T, C)
        x = x + out
        
        info = {
            "winners": winners,
            "costs": costs,
            "difficulty": difficulty,
            "affinity": affinity,
            "recon_loss": total_recon_loss # 把 Bridge Loss 传出去
        }
        
        return x, v_first, info


class CaMoE_System(nn.Module):
    """
    CaMoE v10.0 完整系统
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.num_rwkv_experts = config.get('num_rwkv_experts', 2)
        self.num_experts = self.num_rwkv_experts + 1
        self.vocab_size = config['vocab_size']
        self.gradient_checkpointing = False
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.bridge = UltimateBridge(self.n_embd, config.get('prefix_len', 16))
        
        self.blocks = nn.ModuleList()
        for i in range(self.n_layer):
            self.blocks.append(CaMoE_Block(
                self.n_embd,
                self.n_layer,
                i,
                config['head_size'],
                self.num_rwkv_experts,
                self.bridge
            ))
        
        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Market
        self.capital_manager = CapitalManager(
            self.n_layer, self.num_experts,
            total_capital=config.get('total_capital', 10000.0),
            min_share=config.get('min_capital_share', 0.05),
            tax_threshold=config.get('tax_threshold', 1.5),
            tax_rate=config.get('tax_rate', 0.15)
        )
        self.router = SparseRouter()
        self.eureka = EurekaController()

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def to(self, device):
        super().to(device)
        self.capital_manager.to(device)
        return self
    
    def forward(self, idx: torch.Tensor, step: int = 0, 
                phase: str = "normal") -> Tuple[torch.Tensor, Dict]:
        x = self.emb(idx)
        v_first = None
        
        use_market = (phase == "normal")
        all_info = {"winners": [], "costs": [], "difficulties": [], "affinities": [],"infos":[]}
        
        for i, block in enumerate(self.blocks):
            shares = self.capital_manager.get_shares(i)
            
            # Eureka
            eureka_override = None
            if use_market and self.training:
                with torch.no_grad():
                    dummy_confs = torch.stack([
                        exp.get_confidence(block.ln2(x)) 
                        for exp in block.experts
                    ], dim=-1)
                    trigger = self.eureka.should_trigger(
                        dummy_confs, step, self.config.get('warmup_steps', 2000)
                    )
                    eureka_override = self.eureka.select_underdog(shares, trigger)
            
            if self.gradient_checkpointing and self.training:
                # 这一行是告诉 PyTorch: "这一层的中间结果别存了，反向传播时再算一遍"
                # block 是你的 CaMoE_Block 实例
                
                # 注意：checkpoint 要求输入必须有 requires_grad=True 的 Tensor
                # x 通常是有梯度的。但如果 Embedding 被冻结可能没有，这里 x 应该是有的。
                
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                # 调用 checkpoint
                # 注意参数顺序要和 CaMoE_Block.forward 完全一致
                x, v_first, info = checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, 
                    v_first, 
                    shares, 
                    self.router, 
                    use_market, 
                    eureka_override, 
                    self.training,
                    use_reentrant=False # 推荐设置为 False
                )
            else:
                # 正常的前向传播
                x, v_first, info = block(
                    x, v_first, shares, self.router, use_market, eureka_override, self.training)
            
            all_info["winners"].append(info["winners"])
            all_info["costs"].append(info["costs"])
            all_info["difficulties"].append(info["difficulty"])
            all_info["affinities"].append(info["affinity"])
        
        logits = self.head(self.ln_out(x))
        return logits, all_info
    
    def compute_losses(self, logits, targets, all_info):
        B, T = targets.shape
        main_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        
        with torch.no_grad():
            token_losses = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction='none'
            ).reshape(B, T)
        
        critic_loss = 0.0
        for i, diff in enumerate(all_info.get("difficulties", [])):
            baseline = self.capital_manager.baseline_losses[i]
            target = F.relu(token_losses - baseline)
            critic_loss += F.smooth_l1_loss(diff.squeeze(-1), target)
        
        if len(all_info.get("difficulties", [])) > 0:
            critic_loss /= len(all_info["difficulties"])

        bridge_loss = 0.0
        count = 0

        for info in all_info.get("infos", []): # 假设 forward 返回了这个
             if "recon_loss" in info and isinstance(info["recon_loss"], torch.Tensor):
                 bridge_loss += info["recon_loss"]
                 count += 1
        
        if count > 0:
            bridge_loss /= count

        total_loss = main_loss + 0.1 * critic_loss + 0.05 * bridge_loss
        return total_loss, token_losses, main_loss, critic_loss
    
    def update_market(self, all_info, token_losses, step):
        with torch.no_grad():
            for i in range(self.n_layer):
                if i >= len(all_info.get("winners", [])):
                    continue
                
                self.capital_manager.update(
                    i,
                    all_info["winners"][i],
                    token_losses,
                    all_info["costs"][i],
                    all_info["difficulties"][i]
                )
                
                baseline = self.capital_manager.baseline_losses[i].item()
                self.blocks[i].critic.settle(
                    all_info["affinities"][i],
                    all_info["winners"][i],
                    token_losses,
                    baseline
                )
    
    def log_market_health(self) -> Dict:
        metrics = {}
        for i in [0, self.n_layer // 2, self.n_layer - 1]:
            caps = self.capital_manager.capitals[i]
            shares = caps / caps.sum() * 100
            
            sorted_caps, _ = torch.sort(caps)
            n = self.num_experts
            idx = torch.arange(1, n + 1, device=caps.device, dtype=caps.dtype)
            gini = ((2 * idx - n - 1) * sorted_caps).sum() / (n * caps.sum() + 1e-6)
            
            metrics[f"L{i}/TransShare"] = shares[-1].item()
            metrics[f"L{i}/Gini"] = gini.item()
            metrics[f"L{i}/CriticCap"] = self.blocks[i].critic.capital.item()
        
        return metrics