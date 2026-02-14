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

# ==================== CUDA Kernel ====================
HEAD_SIZE = 64
CHUNK_LEN = 16
USE_CUDA = False
RUN_CUDA_RWKV7 = None

def init_rwkv7_cuda():
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
                return y
            
            @staticmethod
            def backward(ctx, dy):
                r, w, k, v, a, b, s, sa = ctx.saved_tensors
                dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
                torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
                return dr, dw, dk, dv, da, db
        
        def _run_cuda(r, w, k, v, a, b):
            B, T, HC = r.shape
            H = HC // HEAD_SIZE
            
            # View 变换
            # 关键改动：不再强制转 .float() (FP32)
            # 保持原有的 dtype (BF16)，只做 contiguous 和 view
            r, w, k, v, a, b = [
                x.contiguous().view(B, T, H, HEAD_SIZE) 
                for x in [r, w, k, v, a, b]
            ]
            
            out = RWKV7_ClampW.apply(r, w, k, v, a, b)
            out = out.view(B, T, HC)
            return out
        
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
        
        # 初始化权重
        with torch.no_grad():
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()
    
    def forward(self, x: torch.Tensor, v_first: torch.Tensor = None):
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
        x_att = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk * a)
        
            
        state_representation = x_att.clone() 
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


class DeepEmbedAttention(nn.Module):
    """
    DeepEmbedAttention (DEA)：与 TimeMix 并行的因果 Self-Attention 分支。
    - Q/K/V 经 DeepEmbed（k_emb, v_emb 按 token id 查表）调制
    - Q/K/V 做 token-shift（与上一位置混合）
    - 使用 soft-cap: 64 * tanh(scores / 1024) 再 softmax
    """
    
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size: int, vocab_size: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.n_head = n_embd // head_size
        self.head_size = head_size
        C = n_embd
        H = self.n_head
        
        # Q/K/V 投影
        self.qq = nn.Linear(C, C, bias=False)
        self.k1 = nn.Linear(C, C, bias=False)
        self.k2 = nn.Linear(C, C, bias=False)
        self.v1 = nn.Linear(C, C, bias=False)
        self.v2 = nn.Linear(C, C, bias=False)
        
        # DeepEmbed：按 token id 查表，调制 K/V
        self.k_emb = nn.Parameter(torch.zeros(vocab_size, C))
        self.v_emb = nn.Parameter(torch.zeros(vocab_size, C))
        nn.init.normal_(self.k_emb, std=0.02)
        nn.init.normal_(self.v_emb, std=0.02)
        
        # Token-shift 系数（与 backbone time_shift 类似）
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / max(n_layer, 1))
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / max(C - 1, 1)
            self.x_q = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        
        self.ln_q = nn.LayerNorm(C)
        self.ln_k = nn.LayerNorm(C)
        self.ln_v = nn.LayerNorm(C)
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            self.qq.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.k1.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.k2.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.v1.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.v2.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
    
    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C), idx: (B, T) token ids
        return: (B, T, C) DEA 分支输出
        """
        B, T, C = x.shape
        # Q
        q = self.qq(x)
        q_prev = self.time_shift(q) - q
        q = q + q_prev * self.x_q
        q = self.ln_q(q)
        
        # K: k1 -> k2 -> * k_emb[idx]
        k = self.k1(x)
        k = self.k2(k) * self.k_emb[idx]
        k_prev = self.time_shift(k) - k
        k = k + k_prev * self.x_k
        k = self.ln_k(k)
        
        # V: v1 -> v2 -> tanh -> * v_emb[idx]
        v = self.v1(x)
        v = torch.tanh(self.v2(v)) * self.v_emb[idx]
        v_prev = self.time_shift(v) - v
        v = v + v_prev * self.x_v
        v = self.ln_v(v)
        
        # Causal self-attention with soft-cap (from temp.py)
        scale = 1024.0
        cap_scale = 64.0
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = cap_scale * torch.tanh(scores)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = attn @ v
        return out