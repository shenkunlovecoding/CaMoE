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
CUDA_KERNEL_EXPECTS_FP32 = False
W_SCALE = -math.exp(-0.5)

def init_rwkv7_cuda():
    r"""init_rwkv7_cuda() -> None

    编译并注册 RWKV7 自定义 CUDA 算子。

    成功后会设置全局 ``USE_CUDA=True`` 并绑定 ``RUN_CUDA_RWKV7``。
    """
    global USE_CUDA, RUN_CUDA_RWKV7, CUDA_KERNEL_EXPECTS_FP32
    
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
        use_fast_math = os.environ.get("CAMOE_DISABLE_FAST_MATH", "0") != "1"
        force_fp32_kernel = os.environ.get("CAMOE_FORCE_FP32_KERNEL", "0") == "1"

        flags = [
            '-res-usage', 
            f'-D_N_={HEAD_SIZE}', 
            f"-D_CHUNK_LEN_={CHUNK_LEN}", 
            "-O3",  
            "--extra-device-vectorization"
        ]
        if use_fast_math:
            flags.append("--use_fast_math")
        if force_fp32_kernel:
            flags.append("-D_FP32_")
        cxx_flags = ["-D_FP32_"] if force_fp32_kernel else []
        print(f"🔧 RWKV CUDA flags: fast_math={use_fast_math}, fp32_kernel={force_fp32_kernel}")
        
        ext_name = "rwkv7_clampw"
        load(
            name=ext_name, 
            sources=[cu_file, cpp_file], 
            is_python_module=False, 
            verbose=True, 
            extra_cuda_cflags=flags,
            extra_cflags=cxx_flags,
        )
        # TORCH_LIBRARY namespace is fixed in rwkv7_clampw.cpp
        op_ns = torch.ops.rwkv7_clampw
        
        class RWKV7_ClampW(torch.autograd.Function):
            @staticmethod
            def forward(ctx, r, w, k, v, a, b):
                B, T, H, C = r.shape
                assert T % CHUNK_LEN == 0
                expected_dtype = torch.float32 if force_fp32_kernel else torch.bfloat16
                assert all(i.dtype == expected_dtype for i in [r, w, k, v, a, b]), "RWKV7 kernel dtype mismatch"
                assert all(i.is_contiguous() for i in [r, w, k, v, a, b]), "RWKV7 kernel needs contiguous tensors"
                
                # 创建输出和状态张量
                # 注意：BF16 Kernel 要求输入输出是 BF16/FP16，但 State 是 FP32
                y = torch.empty_like(v)
                s = torch.empty(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=r.device)
                sa = torch.empty(B, T, H, C, dtype=torch.float32, device=r.device)
                
                op_ns.forward(r, w, k, v, a, b, y, s, sa)
                ctx.save_for_backward(r, w, k, v, a, b, s, sa)
                return y,sa
            
            @staticmethod
            def backward(ctx, dy):
                expected_dtype = torch.float32 if force_fp32_kernel else torch.bfloat16
                assert dy.dtype == expected_dtype, "RWKV7 backward dtype mismatch"
                assert dy.is_contiguous(), "RWKV7 backward needs contiguous dy"
                r, w, k, v, a, b, s, sa = ctx.saved_tensors
                dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
                op_ns.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
                return dr, dw, dk, dv, da, db
        
        def _run_cuda(r, w, k, v, a, b):
            B, T, HC = r.shape
            H = HC // HEAD_SIZE
            orig_dtype = r.dtype
            target_dtype = torch.float32 if CUDA_KERNEL_EXPECTS_FP32 else torch.bfloat16
            
            # View 变换
            # 关键改动：不再强制转 .float() (FP32)
            # 保持原有的 dtype (BF16)，只做 contiguous 和 view
            r, w, k, v, a, b = [
                x.to(dtype=target_dtype).contiguous().view(B, T, H, HEAD_SIZE)
                for x in [r, w, k, v, a, b]
            ]
            
            out , sa = RWKV7_ClampW.apply(r, w, k, v, a, b)
            out = out.view(B, T, HC).to(orig_dtype)
            sa = sa.view(B, T, HC).to(orig_dtype)
            return out, sa
        
        RUN_CUDA_RWKV7 = _run_cuda
        USE_CUDA = True
        CUDA_KERNEL_EXPECTS_FP32 = force_fp32_kernel
        print(
            f"✅ RWKV-7 CUDA Kernel (ClampW + {'FP32' if force_fp32_kernel else 'BF16'}) Ready"
        )
        
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
        self.ln_sa = nn.LayerNorm(n_embd)
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

    @staticmethod
    def _report_nonfinite(x: torch.Tensor, name: str, layer_id: int) -> None:
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
                f"❌ NaNDebug-TimeMix | layer={layer_id} | tensor={name} | "
                f"bad={bad_count}/{total} | finite_min={vmin:.6e} | finite_max={vmax:.6e}"
            )

    @staticmethod
    def _run_torch_fallback(
        r: torch.Tensor,
        w_raw: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        n_head: int,
        head_size: int,
    ) -> torch.Tensor:
        r"""纯 PyTorch 的 TimeMix 递推实现，用于 CUDA kernel 异常时的应急路径。"""
        B, T, C = r.shape
        H, N = n_head, head_size
        r = r.view(B, T, H, N).float()
        w_raw = w_raw.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()

        # 与 CUDA kernel 保持同样的 w 变换
        w = torch.exp(W_SCALE / (1.0 + torch.exp(-w_raw)))

        state = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float32)
        ys = []
        for t in range(T):
            rt = r[:, t]  # [B,H,N]
            wt = w[:, t]
            kt = k[:, t]
            vt = v[:, t]
            at = a[:, t]
            bt = b[:, t]

            sa = (state * at.unsqueeze(-2)).sum(dim=-1)  # [B,H,N_i]
            state = (
                state * wt.unsqueeze(-2)
                + sa.unsqueeze(-1) * bt.unsqueeze(-2)
                + vt.unsqueeze(-1) * kt.unsqueeze(-2)
            )
            yt = (state * rt.unsqueeze(-2)).sum(dim=-1)  # [B,H,N_i]
            ys.append(yt)

        y = torch.stack(ys, dim=1).reshape(B, T, C)
        return y.to(r.dtype)
    
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
        use_fallback = os.environ.get("CAMOE_FORCE_TIMEMIX_FALLBACK", "0") == "1"
        if (not use_fallback) and USE_CUDA and RUN_CUDA_RWKV7 is not None:
            x_att , sa = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk * a)
            state_representation = self.ln_sa(sa)
        else:
            x_att = self._run_torch_fallback(r, w, k, v, -kk, kk * a, self.n_head, self.head_size)
            state_representation = x_att.clone() 
        if os.environ.get("CAMOE_NAN_DEBUG", "0") == "1":
            self._report_nonfinite(x_att, "x_att_raw", self.layer_id)
        if os.environ.get("CAMOE_SANITIZE_TIMEMIX_OUT", "0") == "1":
            x_att = torch.nan_to_num(x_att, nan=0.0, posinf=0.0, neginf=0.0)

            
        
        x_att = self.ln_x(x_att.view(B * T, C)).view(B, T, C)
        if os.environ.get("CAMOE_NAN_DEBUG", "0") == "1":
            self._report_nonfinite(x_att, "x_att_ln", self.layer_id)
        
        # Bonus term
        x_att = x_att + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)
        ).view(B, T, C)
        
        out = self.output(x_att * g)
        if os.environ.get("CAMOE_NAN_DEBUG", "0") == "1":
            self._report_nonfinite(out, "att_out", self.layer_id)
        
        return out, v_first , state_representation

