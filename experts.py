import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseRWKVFFN(nn.Module):
    """
    RWKV FFN 专家 (无状态版)
    """
    def __init__(self, n_embd: int, expand: int = 4):
        super().__init__()
        hidden = n_embd * expand
        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)
        
        self.confidence = nn.Sequential(
            nn.Linear(n_embd, 64),
            nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        # Init
        self.value.weight.data.zero_()
        nn.init.orthogonal_(self.key.weight.data, gain=2.0)
    
    def get_confidence(self, x):
        return self.confidence(x).squeeze(-1)
        
    def forward(self, x, rwkv_state=None):
        k = torch.relu(self.key(x)) ** 2
        out = self.value(k)
        return out, None

class LinearTransformerExpert(nn.Module):
    """
    Linear Transformer 专家
    """
    def __init__(self, n_embd: int, n_head: int, bridge: nn.Module):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.bridge = bridge 
        
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.o = nn.Linear(n_embd, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, n_embd)
        
        self.confidence = nn.Sequential(
            nn.Linear(n_embd, 64),
            nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        
        # [优化] 初始化：防止初期梯度爆炸
        nn.init.orthogonal_(self.o.weight, gain=0.1)

    def get_confidence(self, x):
        return self.confidence(x).squeeze(-1)

    def forward(self, x, rwkv_state):
        N, C = x.shape
        H, D = self.n_head, self.head_dim
        
        # 1. Bridge
        prefix, recon_loss = self.bridge(x, rwkv_state, return_loss=True)
        P = prefix.shape[1]
        
        # 2. Linear Attention
        # reshape 必须用，防止不连续报错
        q = self.q(x).reshape(N, 1, H, D)
        k = self.k(prefix).reshape(N, P, H, D)
        v = self.v(prefix).reshape(N, P, H, D)
        
        # 3. SDPA (Memory Efficient)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), # [N, H, 1, D]
            k.transpose(1, 2), # [N, H, P, D]
            v.transpose(1, 2), # [N, H, P, D]
            is_causal=False
        )
        
        out = out.transpose(1, 2).reshape(N, C)
        
        # Gating
        g = torch.sigmoid(self.gate(x))
        out = self.o(out) * g
        
        return out, recon_loss