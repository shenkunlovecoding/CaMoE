"""
RWKV-7 TimeMix Backbone (Turbo Edition with ClampW Kernel)
支持 BF16 原生加速
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import math
import os
from typing import Tuple

# ==================== CUDA Kernel ====================
HEAD_SIZE = 64
CHUNK_LEN = 16
USE_CUDA = False
RUN_CUDA_RWKV7 = None

def init_rwkv7_cuda():
    r"""init_rwkv7_cuda() -> None

    编译并注册 RWKV7 自定义 CUDA 算子。

    成功后会设置全局 ``USE_CUDA=True`` 并绑定 ``RUN_CUDA_RWKV7``。
    """
    global USE_CUDA, RUN_CUDA_RWKV7
    
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_dir = os.path.join(curr_dir, 'cuda')
        
        # 新文件名
        cu_file = os.path.join(cuda_dir, 'rwkv7_clampw.cu')
        cpp_file = os.path.join(cuda_dir, 'rwkv7_clampw.cpp')
        
        if not (os.path.exists(cu_file) and os.path.exists(cpp_file)):
            print(f"⚠️ CUDA files not found: {cu_file}")
            print("   Using Fallback (Slow FP32)...")
            return
        
        # 编译标志：如果不加 -D_FP32_，默认就是 BF16
        # 注意：HEAD_SIZE 和 CHUNK_LEN 依然要传
        flags = [
            '-res-usage', 
            f'-D_C_={HEAD_SIZE}', 
            f"-D_CHUNK_LEN_={CHUNK_LEN}", 
            "--use_fast_math", 
            "-O3", 
            "-Xptxas -O3", 
            "--extra-device-vectorization"
        ]
        
        load(
            name="rwkv7_clampw", 
            sources=[cu_file, cpp_file], 
            is_python_module=False, 
            verbose=True, 
            extra_cuda_cflags=flags
        )
        
        class RWKV7_ClampW(torch.autograd.Function):
            @staticmethod
            def forward(ctx, r, w, k, v, a, b):
                B, T, H, C = r.shape
                assert T % CHUNK_LEN == 0
                
                # 创建输出和状态张量
                # 注意：BF16 Kernel 要求输入输出是 BF16/FP16，但 State 是 FP32
                y = torch.empty_like(v)
                s = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=r.device)
                sa = torch.empty(B, T, H, C, dtype=torch.float32, device=r.device)
                
                torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, y, s, sa)
                ctx.save_for_backward(r, w, k, v, a, b, s, sa)
                return y, sa
            
            @staticmethod
            def backward(ctx, dy):
                r, w, k, v, a, b, s, sa = ctx.saved_tensors
                dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
                torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
                return dr, dw, dk, dv, da, db
        
        def _run_cuda(r, w, k, v, a, b):
            B, T, HC = r.shape
            H = HC // HEAD_SIZE
            orig_dtype = r.dtype
            
            # View 变换
            # 关键改动：不再强制转 .float() (FP32)
            # 保持原有的 dtype (BF16)，只做 contiguous 和 view
            r, w, k, v, a, b = [
                x.contiguous().view(B, T, H, HEAD_SIZE) 
                for x in [r, w, k, v, a, b]
            ]
            
            out, sa = RWKV7_ClampW.apply(r, w, k, v, a, b)
            out = out.view(B, T, HC)
            sa = sa.view(B, T, HC).to(orig_dtype)
            return out, sa
        
        RUN_CUDA_RWKV7 = _run_cuda
        USE_CUDA = True
        print("✅ RWKV-7 CUDA Kernel (ClampW + BF16) Ready")
        
    except Exception as e:
        print(f"⚠️ CUDA init failed: {e}")


# ==================== RWKV-7 TimeMix ====================
class RWKV7_TimeMix(nn.Module):
    """
    RWKV-7 TimeMix (Attention equivalent)
    完整实现动态状态演化
    """
    
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size: int = 64):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.head_size = head_size
        self.n_head = n_embd // head_size
        self.n_embd = n_embd
        
        assert n_embd % head_size == 0
        
        C = n_embd
        H = self.n_head
        N = head_size
        
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            
            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                return x
            
            # 按RWKV-7规范设置LORA维度
            D_DECAY_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
            D_AAA_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
            D_MV_LORA = max(32, int(round((1.7 * (C ** 0.5)) / 32) * 32))
            D_GATE_LORA = max(32, int(round((5 * (C ** 0.5)) / 32) * 32))
            
            # Decay
            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            
            # AAA (in-context learning rate)
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            
            # Value residual
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            
            # Gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))
            
            # Key normalization
            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)
        self.ln_sa = nn.LayerNorm(n_embd)
        
        # 初始化权重
        with torch.no_grad():
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()
    
    def forward(
        self, x: torch.Tensor, v_first: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""forward(x, v_first=None) -> Tuple[Tensor, Tensor, Tensor]

        执行 RWKV-7 TimeMix 主干前向。

        Args:
          x (Tensor): 形状 ``[B, T, C]``。
          v_first (Tensor, optional): 首层 value 缓存。Default: ``None``。

        Returns:
          Tuple[Tensor, Tensor, Tensor]:
          输出 ``out``、更新后的 ``v_first``、以及中间状态 ``state_representation``。
        """
        B, T, C = x.size()
        H = self.n_head
        N = self.head_size
        
        # Token shift
        xx = self.time_shift(x) - x
        
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        # Value residual (layer 0 stores, others use)
        if self.layer_id == 0:
            v_first = v.clone()
        else:
            if v_first is not None:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        
        # RWKV-7 动态状态演化
        assert USE_CUDA and RUN_CUDA_RWKV7 is not None
        x_att, sa = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk * a)
        
        state_representation = self.ln_sa(sa)
        x_att = self.ln_x(x_att.view(B * T, C)).view(B, T, C)
        
        # Bonus term
        x_att = x_att + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)
        ).view(B, T, C)
        
        out = self.output(x_att * g)
        
        return out, v_first , state_representation


# ==================== DeepEmbed + DeepEmbedAttention (from RWKV-7 Goose) ====================
# 思想：用「词表级」嵌入对 K/V 做逐 token 调制，并增加一条因果 Self-Attention 分支（Q/K/V token-shift + soft-cap）


class SharedDeepEmbed(nn.Module):
    """
    全模型共享的 DeepEmbed 表。
    通过跨层共享，将词表参数从 O(n_layer * vocab * dim) 降为 O(vocab * dim)。
    """

    def __init__(self, vocab_size: int, k_dim: int, v_dim: int):
        super().__init__()
        self.k_emb = nn.Embedding(vocab_size, k_dim)
        self.v_emb = nn.Embedding(vocab_size, v_dim)
        nn.init.normal_(self.k_emb.weight, std=0.02)
        nn.init.normal_(self.v_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor):
        r"""forward(idx) -> Tuple[Tensor, Tensor]

        Args:
          idx (Tensor): 形状 ``[B, T]`` 的 token id。

        Returns:
          Tuple[Tensor, Tensor]: ``k_emb(idx)`` 与 ``v_emb(idx)``。
        """
        return self.k_emb(idx), self.v_emb(idx)


class DeepEmbedAttention(nn.Module):
    """
    DeepEmbedAttention (DEA)：与 TimeMix 并行的因果 Self-Attention 分支。
    - Q/K/V 经 DeepEmbed（k_emb, v_emb 按 token id 查表）调制
    - Q/K/V 做 token-shift（与上一位置混合）
    - 使用 soft-cap: 64 * tanh(scores / 1024) 再 softmax
    """
    
    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        head_size: int,
        vocab_size: int,
        shared_deep_embed: nn.Module = None,
        q_dim: int = 256,
        kv_dim: int = 32,
        score_scale: float = 1024.0,
        cap_scale: float = 64.0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.q_dim = min(q_dim, n_embd)
        self.kv_dim = min(kv_dim, self.q_dim)
        self.score_scale = float(score_scale)
        self.cap_scale = float(cap_scale)
        C = n_embd

        # RWKV-8 风格低维路径：Q(C->q_dim), K/V(C->kv_dim->up)
        self.qq = nn.Linear(C, self.q_dim, bias=False)
        self.k = nn.Linear(C, self.kv_dim, bias=False)
        self.k_up = nn.Linear(self.kv_dim, self.q_dim, bias=False)
        self.v = nn.Linear(C, self.kv_dim, bias=False)
        self.v_up = nn.Linear(self.kv_dim, C, bias=False)

        # DeepEmbed：默认使用跨层共享表；未提供则回退为层内表
        self.shared_deep_embed = shared_deep_embed
        if self.shared_deep_embed is None:
            self.k_emb = nn.Embedding(vocab_size, self.q_dim)
            self.v_emb = nn.Embedding(vocab_size, C)
            nn.init.normal_(self.k_emb.weight, std=0.02)
            nn.init.normal_(self.v_emb.weight, std=0.02)
        else:
            self.k_emb = None
            self.v_emb = None
        
        # Token-shift 系数（与 backbone time_shift 类似）
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / max(n_layer, 1))
            ddd_q = torch.ones(1, 1, self.q_dim)
            ddd_v = torch.ones(1, 1, C)
            for i in range(self.q_dim):
                ddd_q[0, 0, i] = i / max(self.q_dim - 1, 1)
            for i in range(C):
                ddd_v[0, 0, i] = i / max(C - 1, 1)
            self.x_q = nn.Parameter(1.0 - torch.pow(ddd_q, 0.2 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd_q, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd_v, 0.7 * ratio_1_to_almost0))

        self.ln_q = nn.LayerNorm(self.q_dim)
        self.ln_k = nn.LayerNorm(self.q_dim)
        self.ln_v = nn.LayerNorm(C)
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            self.qq.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.k.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.k_up.weight.data.uniform_(-0.05 / (self.kv_dim ** 0.5), 0.05 / (self.kv_dim ** 0.5))
            self.v.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.v_up.weight.data.uniform_(-0.05 / (self.kv_dim ** 0.5), 0.05 / (self.kv_dim ** 0.5))
    
    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""forward(x, idx) -> Tensor

        Args:
          x (Tensor): 形状 ``[B, T, C]``。
          idx (Tensor): 形状 ``[B, T]`` 的 token id。

        Returns:
          Tensor: 形状 ``[B, T, C]`` 的 DEA 分支输出。
        """
        B, T, C = x.shape
        if self.shared_deep_embed is not None:
            k_emb, v_emb = self.shared_deep_embed(idx)
        else:
            k_emb = self.k_emb(idx)
            v_emb = self.v_emb(idx)

        # Q
        q = self.qq(x)
        q_prev = self.time_shift(q) - q
        q = q + q_prev * self.x_q
        q = self.ln_q(q)
        
        # K: C -> kv_dim -> q_dim，再由 token 级 DeepEmbed 调制
        k = self.k_up(self.k(x)) * k_emb
        k_prev = self.time_shift(k) - k
        k = k + k_prev * self.x_k
        k = self.ln_k(k)
        
        # V: C -> kv_dim -> C，保留 tanh 非线性后调制
        v = torch.tanh(self.v_up(self.v(x))) * v_emb
        v_prev = self.time_shift(v) - v
        v = v + v_prev * self.x_v
        v = self.ln_v(v)
        
        # Causal self-attention with soft-cap
        scores = (q @ k.transpose(-2, -1)) / self.score_scale
        scores = self.cap_scale * torch.tanh(scores)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = attn @ v
        return out
