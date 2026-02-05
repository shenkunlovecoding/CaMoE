import torch
import torch.nn as nn
import torch.nn.functional as F

class UltimateBridge(nn.Module):
    def __init__(self, n_embd: int, max_prefix_len: int = 64):
        super().__init__()
        self.max_prefix_len = max_prefix_len
        self.n_embd = n_embd
        
        # 1. 压缩/融合层
        # [N, 2C] -> [N, 2C]
        self.compressor = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd * 2),
            nn.LayerNorm(n_embd * 2),
            nn.GELU()
        )
        
        # 2. 生成层 (Low-Rank 抽脂手术)
        # 原来是: Linear(2C, 64*C) -> 75M 参数 ❌
        # 现在是: Linear(2C, 64*32) + Linear(32, C) -> 2M 参数 ✅
        self.low_rank_dim = 32
        
        # 第一步：生成低维特征 [N, 64 * 32]
        self.generator_low = nn.Linear(n_embd * 2, max_prefix_len * self.low_rank_dim)
        
        # 第二步：上采样回 C 维 [32 -> C] (参数共享)
        self.upsample = nn.Linear(self.low_rank_dim, n_embd, bias=False)
        
        # 3. 内部位置编码
        # [1, 64, C]
        self.pos_emb = nn.Parameter(torch.zeros(1, max_prefix_len, n_embd))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
        self.norm = nn.LayerNorm(n_embd)

        # 4. 辅助重构 Loss 头 (Low-Rank 抽脂手术)
        # 原来是: Linear(64*C, C) -> 37M 参数 ❌
        # 现在是: MeanPool + Linear(C, C) -> 0.6M 参数 ✅
        # 逻辑：与其暴力全连接，不如先把 Prefix 平均一下，再尝试还原 x
        self.decoder_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor, rwkv_state: torch.Tensor, return_loss: bool = False):
        """
        x: [N, C]
        rwkv_state: [N, C]
        """
        N, C = x.shape
        
        # 1. 融合状态
        combined = torch.cat([x, rwkv_state.detach()], dim=-1) 
        feat = self.compressor(combined)
        
        # 2. Low-Rank 生成 Prefix
        # [N, 2C] -> [N, 64, 32]
        low_feat = self.generator_low(feat).reshape(N, self.max_prefix_len, self.low_rank_dim)
        # [N, 64, 32] -> [N, 64, C]
        prefix = self.upsample(low_feat)
        
        # 3. 注入位置信息
        prefix = prefix + self.pos_emb
        prefix = self.norm(prefix)
        
        recon_loss = 0.0
        if return_loss:
            # [修改] 更高效的重构 Loss
            # 我们不指望能完美还原 x，只要 Prefix 的平均值包含了 x 的信息即可
            # Pooling: [N, 64, C] -> [N, C]
            prefix_summary = prefix.mean(dim=1)
            recon_x = self.decoder_proj(prefix_summary)
            recon_loss = F.mse_loss(recon_x, x.detach())
            
        return prefix, recon_loss