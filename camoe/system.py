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

from .backbone import RWKV7_TimeMix, DeepEmbedAttention, SharedDeepEmbed
from .bridge import UltimateBridge
from .experts import SparseRWKVFFN, LinearTransformerExpert
from .critic import CriticVC
from .market import CapitalManager, SparseRouter

class CaMoE_Block(nn.Module):
    r"""单个 CaMoE Block，包含 TimeMix、DEA 与 Top-2 专家路由。"""
    
    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        head_size: int,
        config: Dict,
        bridge: nn.Module,
        shared_deep_embed: nn.Module = None,
    ) -> None:
        r"""初始化单层 CaMoE Block。"""
        super().__init__()
        
        self.layer_id = layer_id
        self.num_rwkv = config.get('num_rwkv_experts', 6)
        self.num_trans = config.get('num_trans_experts', 2)
        self.num_experts = self.num_rwkv + self.num_trans
        self.n_embd = n_embd
        self.bridge = bridge
        self.nan_debug = config.get("nan_debug", False)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # RWKV-7 TimeMix (Backbone)
        self.att = RWKV7_TimeMix(n_embd, n_layer, layer_id, head_size)
        
        # DeepEmbedAttention (v18.5-test): 与 TimeMix 并行的因果 Attention 分支
        self.use_deep_embed_attention = config.get("use_deep_embed_attention", False)
        vocab_size = config.get("vocab_size", 65536)
        if self.use_deep_embed_attention:
            self.dea = DeepEmbedAttention(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                head_size=head_size,
                vocab_size=vocab_size,
                shared_deep_embed=shared_deep_embed,
                q_dim=config.get("dea_q_dim", 256),
                kv_dim=config.get("dea_kv_dim", 32),
                score_scale=config.get("dea_score_scale", 1024.0),
                cap_scale=config.get("dea_cap_scale", 64.0),
            )
        else:
            self.dea = None
        
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

    def _assert_finite(self, x: torch.Tensor, name: str, step: int) -> None:
        if (not self.nan_debug) or (x is None):
            return
        if not torch.is_floating_point(x):
            return
        if torch.isfinite(x).all():
            return
        with torch.no_grad():
            bad = ~torch.isfinite(x)
            bad_count = int(bad.sum().item())
            total = x.numel()
            finite_x = x[torch.isfinite(x)]
            if finite_x.numel() > 0:
                vmin = float(finite_x.min().item())
                vmax = float(finite_x.max().item())
            else:
                vmin = float("nan")
                vmax = float("nan")
            print(
                f"❌ NaNDebug-Block | step={step} | block={self.layer_id} | tensor={name} | "
                f"bad={bad_count}/{total} | finite_min={vmin:.6e} | finite_max={vmax:.6e}"
            )
        raise RuntimeError(f"NaN/Inf in block {self.layer_id}, tensor={name}, step={step}")
    
    def forward(self, 
                x: torch.Tensor, 
                v_first: torch.Tensor,
                capital_shares: torch.Tensor,
                router: SparseRouter,
                step: int,
                warmup_steps: int,
                use_market: bool = True,
                training: bool = True,
                idx: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        r"""forward(x, v_first, capital_shares, router, step, warmup_steps, use_market=True, training=True, idx=None) -> Tuple[Tensor, Tensor, Dict]

        执行单层前向：并行 TimeMix/DEA、市场路由、专家执行与残差融合。

        Args:
          x (Tensor): 形状 ``[B, T, C]`` 的输入隐藏状态。
          v_first (Tensor): RWKV 首层 value 缓存。
          capital_shares (Tensor): 形状 ``[E]`` 的专家资本占比。
          router (SparseRouter): 路由器实例。
          step (int): 当前训练步。
          warmup_steps (int): warmup 阶段边界。
          use_market (bool, optional): 是否使用市场路由。Default: ``True``。
          training (bool, optional): 是否训练模式。Default: ``True``。
          idx (Tensor, optional): 形状 ``[B, T]`` 的 token id。Default: ``None``。

        Returns:
          Tuple[Tensor, Tensor, Dict]:
          更新后的隐藏状态、``v_first`` 与路由信息字典。
        """
        
        B, T, C = x.shape
        self._assert_finite(x, "x_in", step)
        
        # 1. TimeMix + DEA 并行分支（同一份 pre-norm 输入）
        x_ln = self.ln1(x)
        self._assert_finite(x_ln, "x_ln", step)
        att_out, v_first, rwkv_state = self.att(x_ln, v_first)
        self._assert_finite(att_out, "att_out", step)
        self._assert_finite(v_first, "v_first_att", step)
        self._assert_finite(rwkv_state, "rwkv_state", step)
        if self.dea is not None and idx is not None:
            dea_out = self.dea(x_ln, idx)
            self._assert_finite(dea_out, "dea_out", step)
            x = x + att_out + dea_out
        else:
            x = x + att_out
        self._assert_finite(x, "x_after_att", step)
        
        h = self.ln2(x)
        self._assert_finite(h, "h_ln2", step)
        
        # 2. 计算所有专家的 Confidence
        conf_list = [exp.get_confidence(h) for exp in self.experts]
        confidences = torch.stack(conf_list, dim=-1)  # [B, T, E]
        self._assert_finite(confidences, "confidences", step)
        
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
            self._assert_finite(difficulty, "difficulty", step)
            self._assert_finite(affinity, "affinity", step)
            critic_subsidy = self.critic.apply_to_bids(torch.zeros_like(confidences), affinity)
            self._assert_finite(critic_subsidy, "critic_subsidy", step)
            winners, weights, costs, bids = router.route(
                confidences, capital_shares, difficulty, critic_subsidy, training
            )
            self._assert_finite(weights, "weights", step)
            self._assert_finite(costs, "costs", step)
            self._assert_finite(bids, "bids", step)
        
        # 4. 生成 Bridge Prefix (一次性，供所有 Trans 专家使用)
        flat_h = h.reshape(-1, C)
        flat_state = rwkv_state.reshape(-1, C)
        bridge_prefix = self.bridge(flat_h, flat_state)  # [B*T, P, C]
        self._assert_finite(bridge_prefix, "bridge_prefix", step)
        
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
                self._assert_finite(expert_out, f"expert_out_e{e}", step)
                
                # 加权累加
                weighted_out = expert_out * selected_weights
                self._assert_finite(weighted_out, f"weighted_out_e{e}", step)
                final_out[mask] += weighted_out

        # 残差连接
        self._assert_finite(final_out, "final_out", step)
        x = x + final_out
        self._assert_finite(x, "x_out", step)
        
        info = {
            "winners": winners,
            "costs": costs,
            "difficulty": difficulty,
            "affinity": affinity,
        }
        return x, v_first, info


class CaMoE_System(nn.Module):
    r"""CaMoE 主系统，封装多层 Block、市场状态与损失计算。"""

    def __init__(self, config: Dict) -> None:
        r"""初始化系统级模块与共享组件。"""
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.vocab_size = config['vocab_size']
        self.use_gradient_checkpoint = config.get("use_gradient_checkpoint", True)
        self.nan_debug = config.get("nan_debug", False)
        
        self.num_rwkv_experts = config.get('num_rwkv_experts', 6)
        self.num_trans_experts = config.get('num_trans_experts', 2)
        self.num_experts = self.num_rwkv_experts + self.num_trans_experts
        
        # Embedding
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)

        # Shared DeepEmbed table (optional, recommended for VRAM efficiency)
        self.deep_embed = None
        if config.get("use_deep_embed_attention", False) and config.get("use_shared_deep_embed", True):
            self.deep_embed = SharedDeepEmbed(
                vocab_size=self.vocab_size,
                k_dim=min(config.get("dea_q_dim", 256), self.n_embd),
                v_dim=self.n_embd,
            )
        
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
                bridge=self.bridge,
                shared_deep_embed=self.deep_embed,
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

    def _assert_finite(self, x: torch.Tensor, name: str, step: int, layer_id: int = -1) -> None:
        r"""_assert_finite(x, name, step, layer_id=-1) -> None

        在调试模式下校验张量数值合法性，出现 NaN/Inf 立即抛错并输出定位信息。
        """
        if (not self.nan_debug) or (x is None):
            return
        if not torch.is_floating_point(x):
            return
        if torch.isfinite(x).all():
            return

        with torch.no_grad():
            bad = ~torch.isfinite(x)
            bad_count = int(bad.sum().item())
            total = x.numel()
            finite_x = x[torch.isfinite(x)]
            if finite_x.numel() > 0:
                vmin = float(finite_x.min().item())
                vmax = float(finite_x.max().item())
            else:
                vmin = float("nan")
                vmax = float("nan")
            print(
                f"❌ NaNDebug | step={step} | layer={layer_id} | tensor={name} | "
                f"bad={bad_count}/{total} | finite_min={vmin:.6e} | finite_max={vmax:.6e}"
            )
        raise RuntimeError(f"NaN/Inf detected at step={step}, layer={layer_id}, tensor={name}")
    
    def forward(self, idx: torch.Tensor, step: int = 0, 
                phase: str = "normal") -> Tuple[torch.Tensor, Dict]:
        r"""forward(idx, step=0, phase="normal") -> Tuple[Tensor, Dict]

        执行整网前向并收集各层路由信息。

        Args:
          idx (Tensor): 形状 ``[B, T]`` 的 token id。
          step (int, optional): 当前步数。Default: ``0``。
          phase (str, optional): 训练阶段标签。Default: ``"normal"``。

        Returns:
          Tuple[Tensor, Dict]: ``logits`` 与各层 ``info``。
        """
        x = self.emb(idx)
        self._assert_finite(x, "emb_out", step, -1)
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
            if self.training and self.use_gradient_checkpoint:
                # 开启 Checkpoint 以节省显存
                x, v_first, info = checkpoint(
                    block, 
                    x, v_first, shares, self.router, 
                    step, warmup_steps, use_market, True, idx,
                    use_reentrant=False
                )
            else:
                x, v_first, info = block(
                    x, v_first, shares, self.router,
                    step, warmup_steps, use_market, self.training, idx
                )

            self._assert_finite(x, "block_out", step, i)
            self._assert_finite(v_first, "v_first", step, i)
            self._assert_finite(info["costs"], "costs", step, i)
            self._assert_finite(info["difficulty"], "difficulty", step, i)
            self._assert_finite(info["affinity"], "affinity", step, i)
            
            all_info["winners"].append(info["winners"].detach())
            all_info["costs"].append(info["costs"].detach())
            all_info["difficulties"].append(info["difficulty"].detach())
            all_info["affinities"].append(info["affinity"].detach())
        
        x = self.ln_out(x)
        self._assert_finite(x, "ln_out", step, self.n_layer)
        
        # Output (Tied Embedding Rescale Trick)
        if self.head is not None:
            logits = self.head(x)
        else:
            # Tied embedding 需缩放，避免 logits 幅度过大导致 CE 量纲异常
            x = x * (self.n_embd ** -0.5)
            logits = F.linear(x, self.emb.weight)
        self._assert_finite(logits, "logits", step, self.n_layer)
        
        return logits, all_info
    
    def compute_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        all_info: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        r"""compute_losses(logits, targets, all_info) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]

        计算主损失、token 级损失以及 Critic 损失。

        Args:
          logits (Tensor): 形状 ``[B, T, V]``。
          targets (Tensor): 形状 ``[B, T]``。
          all_info (Dict): 各层难度/路由信息。

        Returns:
          Tuple[Tensor, Tensor, Tensor, Tensor, float]:
          ``total_loss``、``token_losses``、``main_loss``、``critic_loss``、``bridge_loss``。
        """
        if self.config.get("stabilize_logits", False):
            # 训练稳定性保护：避免上游极端值导致 CE 直接 NaN/Inf
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

        B, T = targets.shape
        
        # Main Loss
        main_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        
        # Token Losses (for Market Update)
        with torch.no_grad():
            token_losses = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction='none',
                ignore_index=-100,
            ).reshape(B, T)
            if self.config.get("stabilize_logits", False):
                token_losses = torch.nan_to_num(token_losses, nan=0.0, posinf=100.0, neginf=0.0)
            # ignore_index 位置本身为 0 loss，这里再显式归零，避免后续市场更新误用
            token_losses = token_losses.masked_fill(targets.eq(-100), 0.0)
        
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
    
    def update_market(self, all_info: Dict, token_losses: torch.Tensor, step: int) -> None:
        r"""update_market(all_info, token_losses, step) -> None

        根据 token 级损失结算市场状态与 Critic 资本。

        Args:
          all_info (Dict): 前向收集的市场信息。
          token_losses (Tensor): 形状 ``[B, T]``。
          step (int): 当前训练步。
        """
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
        r"""log_market_health() -> Dict

        汇总所有层的市场健康指标。

        Returns:
          Dict: 包含 RWKV/Transformer 份额、Gini、Critic 资本等指标。
        """
        metrics = {}
        for i in range(self.n_layer):
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
