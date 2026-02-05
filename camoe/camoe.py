"""
CaMoE v12.0 主系统
特性：
1. 混合架构支持：动态配置 RWKV 专家数和 Transformer 专家数 (如 2+2)
2. 共享 Bridge：所有 Transformer 专家共享同一个 Bridge 实例，节省显存并集中训练
3. 持续学习：Bridge 在每一步都计算重构 Loss，无论胜者是谁
4. Eureka 下放：优化计算效率
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple, List
from camoe.backbone import RWKV7_TimeMix
from camoe.bridge import UltimateBridge
from camoe.experts import SparseRWKVFFN, LinearTransformerExpert
from camoe.critic import CriticVC
from camoe.market import CapitalManager, SparseRouter, EurekaController


class CaMoE_Block(nn.Module):
    """
    单个CaMoE块 (支持 N个RWKV + M个Trans 混合架构)
    """
    
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, 
                 head_size: int, config: Dict, bridge: nn.Module):
        super().__init__()
        
        self.layer_id = layer_id
        
        # [架构升级] 动态读取专家数量
        self.num_rwkv = config.get('num_rwkv_experts', 2)
        self.num_trans = config.get('num_trans_experts', 1) # 默认为1，兼容旧配置
        self.num_experts = self.num_rwkv + self.num_trans
        
        self.n_embd = n_embd
        self.bridge = bridge # 接收共享的 Bridge
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # RWKV-7 TimeMix (System 1 永远在线)
        self.att = RWKV7_TimeMix(n_embd, n_layer, layer_id, head_size)
        
        # 专家组构建
        self.experts = nn.ModuleList()
        
        # 1. System 1: RWKV FFN Experts
        for _ in range(self.num_rwkv):
            self.experts.append(SparseRWKVFFN(n_embd))
            
        # 2. System 2: Transformer Experts (共享同一个 Bridge)
        for _ in range(self.num_trans):
            self.experts.append(LinearTransformerExpert(
                n_embd, 
                n_head=n_embd//head_size, 
                bridge=self.bridge 
            ))
        
        # Critic 负责预测总专家数的难度和亲和力
        self.critic = CriticVC(n_embd, self.num_experts)
    
    def forward(self, 
                x: torch.Tensor, 
                v_first: torch.Tensor,
                capital_shares: torch.Tensor,
                router: SparseRouter,
                eureka: EurekaController,
                step: int,
                warmup_steps: int,
                use_market: bool = True,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        B, T, C = x.shape
        
        # 1. TimeMix (Backbone)
        att_out, v_first, rwkv_state = self.att(self.ln1(x), v_first)
        x = x + att_out
        h = self.ln2(x)
        
        # 2. Confidence Calculation (一次性计算所有专家的信心)
        # 注意：LinearTransformerExpert.get_confidence 也只需要 h，不需要 state
        conf_list = [exp.get_confidence(h) for exp in self.experts]
        confidences = torch.stack(conf_list, dim=-1) # [B, T, E]
        
        # 3. Market / Routing
        if not use_market:
            # Inference / No Market: 随机或基于规则，这里保持随机
            winners = torch.randint(0, self.num_experts, (B, T), device=x.device)
            costs = torch.zeros(B, T, device=x.device)
            difficulty = torch.ones(B, T, 1, device=x.device)
            affinity = torch.zeros(B, T, self.num_experts, device=x.device)
        else:
            difficulty, affinity = self.critic(h)
            critic_subsidy = self.critic.apply_to_bids(torch.zeros_like(confidences), affinity)
            winners, costs, bids = router.route(confidences, capital_shares, difficulty, critic_subsidy, training)
            
            # Eureka (扶贫机制)
            if training:
                trigger = eureka.should_trigger(confidences, step, warmup_steps)
                eureka_override = eureka.select_underdog(capital_shares, trigger)
                valid = eureka_override >= 0
                winners = torch.where(valid, eureka_override, winners)
        
        # 4. Bridge 持续学习 (Invisible Training)
        # 无论谁胜出，Bridge 都要尝试重构输入，保持活性
        # 即使这一层全是 RWKV 专家赢了，Bridge 依然在训练，为 Transformer 随时上线做准备
        flat_h = h.reshape(-1, C)
        flat_state = rwkv_state.reshape(-1, C)
        _, recon_loss = self.bridge(flat_h, flat_state, return_loss=True)

        # 5. 专家稀疏执行 (Scatter - Compute - Gather)
        flat_winners = winners.reshape(-1)
        
        # 梯度直通
        flat_conf = confidences.view(-1, self.num_experts)
        row_idx = torch.arange(flat_h.shape[0], device=x.device)
        winning_conf = flat_conf[row_idx, flat_winners].unsqueeze(-1)
        scale_factor = winning_conf / (winning_conf.detach() + 1e-6)
        
        final_out = torch.zeros_like(flat_h)
        
        for e in range(self.num_experts):
            mask_indices = (flat_winners == e).nonzero(as_tuple=True)[0]
            if mask_indices.numel() == 0: continue
            
            # Gather
            selected_h = flat_h[mask_indices]
            selected_scale = scale_factor[mask_indices]
            
            # [关键逻辑] 根据索引判断专家类型
            if e >= self.num_rwkv: 
                # System 2: Transformer Experts (需要 State 通过 Bridge)
                selected_state = flat_state[mask_indices]
                # Expert 内部不再计算 Bridge Loss，只使用 Bridge 输出
                expert_out, _ = self.experts[e](selected_h, selected_state)
            else: 
                # System 1: RWKV FFN Experts (不需要 State)
                expert_out, _ = self.experts[e](selected_h, None)
            
            # Apply Gradient Scaling & Scatter
            expert_out_scaled = expert_out * selected_scale
            expert_out_scaled = expert_out_scaled.to(dtype=final_out.dtype)
            final_out.index_copy_(0, mask_indices, expert_out_scaled)
            
        out = final_out.reshape(B, T, C)
        x = x + out
        
        info = {
            "winners": winners,
            "costs": costs,
            "difficulty": difficulty,
            "affinity": affinity,
            "recon_loss": recon_loss # 全局重构 Loss
        }
        return x, v_first, info


class CaMoE_System(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        
        # [架构升级] 计算总专家数
        num_rwkv = config.get('num_rwkv_experts', 2)
        num_trans = config.get('num_trans_experts', 1)
        self.num_experts = num_rwkv + num_trans
        
        self.vocab_size = config['vocab_size']
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        
        # 共享 Bridge
        self.bridge = UltimateBridge(self.n_embd, config.get('prefix_len', 16))
        
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
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # [关键修复] CapitalManager 现在是 nn.Module，直接作为子模块
        # 它的 buffer 会自动包含在 state_dict() 中
        self.capital_manager = CapitalManager(
            self.n_layer, self.num_experts,
            total_capital=config.get('total_capital', 10000.0),
            min_share=config.get('min_capital_share', 0.05),
            tax_threshold=config.get('tax_threshold', 1.5),
            tax_rate=config.get('tax_rate', 0.15)
        )
        
        # 删掉这几行，不需要了！
        # self.register_buffer("market_capitals", ...)
        # self.register_buffer("market_baselines", ...)
        # self.capital_manager.capitals = ...
        
        self.router = SparseRouter()
        self.eureka = EurekaController()
    
    def forward(self, idx: torch.Tensor, step: int = 0, 
                phase: str = "normal") -> Tuple[torch.Tensor, Dict]:
        x = self.emb(idx)
        v_first = None
        use_market = (phase == "normal")
        
        all_info = {
            "winners": [], "costs": [], "difficulties": [], 
            "affinities": [], "recon_losses": []
        }
        warmup_steps = self.config.get('warmup_steps', 2000)

        for i, block in enumerate(self.blocks):
            shares = self.capital_manager.get_shares(i)
            
            # 调用 Block
            x, v_first, info = block(
                x, v_first, shares, self.router, self.eureka, 
                step, warmup_steps, use_market, self.training
            )
            
            all_info["winners"].append(info["winners"].detach())
            all_info["costs"].append(info["costs"].detach())
            all_info["difficulties"].append(info["difficulty"].detach())
            all_info["affinities"].append(info["affinity"].detach())
            all_info["recon_losses"].append(info["recon_loss"])
        
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

        # Bridge Reconstruction Loss
        bridge_loss = 0.0
        recon_losses = all_info.get("recon_losses", [])
        if recon_losses:
            valid_losses = [l for l in recon_losses if isinstance(l, torch.Tensor)]
            if valid_losses:
                # [修复] 用 sum 代替 stack，更省显存
                bridge_loss = sum(valid_losses) / len(valid_losses)

        total_loss = main_loss + 0.1 * critic_loss + 0.1 * bridge_loss
        
        return total_loss, token_losses, main_loss, critic_loss, bridge_loss
    
    # update_market 和 log_market_health 保持不变...
    def update_market(self, all_info, token_losses, step):
        with torch.no_grad():
            for i in range(self.n_layer):
                if i >= len(all_info.get("winners", [])): continue
                
                # 1. 更新专家资本 (市场机制)
                self.capital_manager.update(
                    i, all_info["winners"][i], token_losses,
                    all_info["costs"][i], all_info["difficulties"][i]
                )
                
                # 2. 更新 Critic 资本 (宏观调控)
                baseline = self.capital_manager.baseline_losses[i].item()
                self.blocks[i].critic.settle(
                    all_info["affinities"][i], all_info["winners"][i],
                    token_losses, baseline
                )

                # [新增] 央行救市 (Bailout) 逻辑
                # 如果 Critic 资本低于 200 (意味着无法有效发放补贴)，强行注入 2000
                if self.blocks[i].critic.capital < 200:
                    self.blocks[i].critic.capital.fill_(2000.0)
                    # 每 100 步打印一次，防止刷屏
                    if step % 100 == 0:
                        print(f"🏛️  Layer {i}: Critic Bailout Triggered (Step {step})")
    
    def log_market_health(self) -> Dict:
        metrics = {}
        check_layers = [0, self.n_layer // 2, self.n_layer - 1]
        check_layers = sorted(list(set([i for i in check_layers if i < self.n_layer])))

        for i in check_layers:
            caps = self.capital_manager.capitals[i]
            # 计算最后两个专家（假设是 Transformer）的份额
            # 如果 num_trans=2, 这里取最后两个
            # 通用写法：
            shares = caps / caps.sum() * 100
            
            # Gini
            sorted_caps, _ = torch.sort(caps)
            n = self.num_experts
            idx = torch.arange(1, n + 1, device=caps.device, dtype=caps.dtype)
            gini = ((2 * idx - n - 1) * sorted_caps).sum() / (n * caps.sum() + 1e-6)
            
            # Log Transformer Total Share
            # 假设 config 还没传进来，这里暂时只打印最后一个作为参考，或者全打
            metrics[f"L{i}/LastExpShare"] = shares[-1].item()
            metrics[f"L{i}/Gini"] = gini.item()
            metrics[f"L{i}/CriticCap"] = self.blocks[i].critic.capital.item()
        
        return metrics