import triton, torch as th
from wind_rosa import load_wind_rosa, wind_rosa, WindRosa


def grad_check(f1, f2, params, backward = True, aux=()):
    if backward: params = [p.clone().requires_grad_() for p in params]
    y1 = f1(*params,*aux)
    y2 = f2(*params,*aux)
    def rel(a,b): return (a-b).norm()/max(b.norm(),1e-30)
    print('Forward rel. error')
    for a,b in zip([y1],[y2]):
        print(f'{rel(a,b):.2e}  ({b.norm():.0e})')

    if not backward: return

    dy = th.randn_like(y1)
    d1 = th.autograd.grad(y1, params, grad_outputs=dy)
    for p in params:
        if p.grad is not None:
            p.grad.random_() # So th.empty doesn't recover the gradient
        p.grad = None
    d2 = th.autograd.grad(y2, params, grad_outputs=dy)
    print('Gradient rel. errors')
    for a,b in zip(d1,d2):
        print(f'{rel(a,b):.2e}  ({b.norm():.0e})')



def benchmark(f, params, backward = True):
    if backward:
        for p in params: p.requires_grad_()
    dy = None
    def wrap():
        y = f(*params)
        if not backward: return
        nonlocal dy
        if dy is None: dy = th.randn_like(y)
        return th.autograd.grad(y, params, grad_outputs=dy)

    wrap() # Warmup
    th.cuda.synchronize()
    th.cuda.reset_peak_memory_stats()
    wrap() # Measure memory
    th.cuda.synchronize()
    print(f'Peak VRAM {th.cuda.max_memory_allocated()/2**30:.2f} GB')
    ms, min_ms, max_ms = triton.testing.do_bench(wrap, quantiles=[0.5,0.2,0.8], warmup=1000,rep=2000)
    print('Time', f'{ms:.2f} ms ({min_ms:.2f} - {max_ms:.2f})')



def naive_fwd(q_, k_, v_):
    y = th.empty_like(v_)
    for bi in range(B):
        for hi in range(H):
            q,k = tuple(q_[bi,:,hi].tolist()),tuple(k_[bi,:,hi].tolist())
            d = {}
            for i in range(len(q)):
                for j in range(i+1):
                    d[k[j:i]] = v_[bi,i,hi]
                j = max(0,i+1-K)
                while not q[j:i+1] in d: j += 1
                y[bi,i,hi] = d[q[j:i+1]]
    return y

def fast_fwd(q, k, v):
    q,k,v = [i.mT.contiguous() for i in [q,k,v]]
    y = th.empty_like(v)
    th.ops.wind_rosa.forward(q,k,v,y, WindRosa.scratch)
    return y.mT


class Naive(th.autograd.Function):
    @staticmethod
    def forward(ctx, q_, k_, v_, C):
        B,T,HC = q_.shape
        assert q_.shape == k_.shape == v_.shape
        assert HC%C == 0
        H = HC//C
        q_,k_,v_ = [((x.view(B,T,H,C)>0)<<th.arange(C,device=q_.device)).sum(dim=-1).to(th.uint8) for x in [q_,k_,v_]]
        y = naive_fwd(q_, k_, v_)
        ctx.save_for_backward(q_,k_,v_)
        ctx.C = C
        return (y[:,:,:,None]>>th.arange(C,device=y.device)&1).reshape(B,T,H*C).float() # .float() needed for pytorch backward graph

    @staticmethod 
    def backward(ctx, dy):
        fwd = fast_fwd # slow_fwd
        q,k,v = ctx.saved_tensors
        C = ctx.C
        base = fwd(q, k, v)
        B,T,H = q.shape
        dq,dk,dv = [th.empty(B,T,H*C, device=dy.device) for x in [q,k,v]]
        for x,dx in [(q,dq),(k,dk),(v,dv)]:
            for ci in range(C):
                for t in range(T):
                    x[:,t,:] ^= 1<<ci
                    off_y = fwd(q, k, v)
                    x[:,t,:] ^= 1<<ci
                    change = sum((((off_y>>j&1).float() - (base>>j&1).float()) * dy[:,:,j::C]).sum(dim=1) for j in range(C))
                    flip_dir = 1 - 2 * (x[:,t,:]>>ci&1).float()
                    dx[:,t,ci::C] = change * flip_dir
        return dq, dk, dv, None


def gen_data(B,T,H,C, binary):
    q,k,v = [th.randn(B,T,H*C,device='cuda') + binary*99*(th.arange(H*C,device='cuda')%C) for i in range(3)] # Hit more edge cases with binary inputs
    return q,k,v


test = 'benchmark'

if test == 'correctness':
    B = 1  # Batch size
    H = 64 # Number of heads
    T = 16 # Sequence length
    C = 8  # Number of bits per channel
    K = 3  # Max match length

    th.manual_seed(0)
    q,k,v = gen_data(B,T,H,C, True)

    load_wind_rosa(T, C, K)

    grad_check(wind_rosa, lambda q,k,v : Naive.apply(q,k,v,C), (q,k,v))

elif test == 'benchmark':
    B = 1     # Batch size
    H = 256   # Number of heads
    T = 2**13 # Sequence length
    C = 8     # Number of bits per channel
    K = 8     # Max match length

    th.manual_seed(0)
    q,k,v = gen_data(B,T,H,C, False)

    load_wind_rosa(T, C, K)

    benchmark(wind_rosa, (q,k,v))
