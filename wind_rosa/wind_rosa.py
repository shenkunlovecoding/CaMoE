import torch as th
from torch.utils.cpp_extension import load

class WindRosa(th.autograd.Function):
    T = C = K = scratch = None
    @staticmethod
    def forward(ctx, q, k, v):
        B,T,HC = q.shape
        C = WindRosa.C
        assert q.shape == k.shape == v.shape
        assert HC%C == 0
        assert T == WindRosa.T
        assert all(x.is_cuda for x in [q,k,v])
        H = HC//C
        q,k,v = [((x.view(B,T,H,C)>0)<<th.arange(C,device=q.device)).sum(dim=-1).to(th.uint8) for x in [q,k,v]]

        # The algorithm requires a lot of temporary scratch space.
        # Share scratch space across layers, can alternatively allocate and free scratch space for every call
        scratch_bytes = th.ops.wind_rosa.scratch_size(B, H)
        if WindRosa.scratch is None or WindRosa.scratch.numel() < scratch_bytes:
            WindRosa.scratch = th.empty(scratch_bytes, dtype=th.uint8, device=q.device)

        q_,k_,v_ = [x.mT.contiguous() for x in [q,k,v]]
        y = th.empty_like(v_)
        th.ops.wind_rosa.forward(q_,k_,v_,y, WindRosa.scratch)
        ctx.save_for_backward(q_,k_,v_)
        return (y.mT[:,:,:,None]>>th.arange(C,device=y.device)&1).reshape(B,T,H*C).float() # .float() needed for pytorch backward graph

    @staticmethod 
    def backward(ctx, dy):
        q_,k_,v_ = ctx.saved_tensors
        B,H,T = q_.shape
        C = WindRosa.C
        assert dy.is_cuda
        assert dy.dtype == th.float32
        assert list(dy.shape) == [B,T,H*C]
        dq,dk,dv = [th.empty(B,H,T,C, device=q_.device) for i in [q_,k_,v_]]
        dy = dy.view(B,T,H,C).transpose(1,2).contiguous()
        th.ops.wind_rosa.backward(q_,k_,v_,dy,dq,dk,dv, WindRosa.scratch)
        dq,dk,dv = [i.transpose(1,2).reshape(B,T,H*C) for i in [dq,dk,dv]]
        return dq, dk, dv

def load_wind_rosa(T, C, K):
    if hasattr(th.ops.wind_rosa, 'forward'): return
    load(name="wind_rosa", sources=['wind_rosa.cu', 'wind_rosa.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=[f'-D_T_={T} -D_C_={C} -D_K_={K}'])
    WindRosa.T = T
    WindRosa.C = C
    WindRosa.K = K
    assert hasattr(th.ops.wind_rosa, 'forward')

def wind_rosa(q, k, v):
    return WindRosa.apply(q, k, v)
