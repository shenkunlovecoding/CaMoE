# Wind ROSA
Rapid Online Suffix Automaton (ROSA) is a symbolic attention mechanism with linearly increasing state size, but constant lookup time per token. QKV ROSA can functionally be implemented as:
```python
# q_,k_,v_ are integers up to the vocabulary size 2^C
# K is the match truncation length
def naive_fwd(q_, k_, v_, K):
    B, T, H = q_.shape
    y = torch.empty_like(v_)
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
```
Since the inputs are discrete, we use finite differences gradients. Specifically, we estimate the gradient as the output changes induced by flipping bits one at a time:
```python
# C is the number of bits in the vocabulary
# K is the match truncation length
def naive_bwd(q, k, v, dy, C, K):
    base = naive_fwd(q, k, v, K)
    B,T,H = q.shape
    dq,dk,dv = [torch.empty(B,T,H*C, device=dy.device) for x in [q,k,v]]
    for x,dx in [(q,dq),(k,dk),(v,dv)]:
        for ci in range(C):
            for t in range(T):
                x[:,t,:] ^= 1<<ci
                off_y = naive_fwd(q, k, v, K)
                x[:,t,:] ^= 1<<ci
                change = sum((((off_y>>j&1).float() - (base>>j&1).float()) * dy[:,:,j::C]).sum(dim=1) for j in range(C))
                flip_dir = 1 - 2 * (x[:,t,:]>>ci&1).float()
                dx[:,t,ci::C] = change * flip_dir
    return dq, dk, dv
```
**Note: While wind_rosa is a mathematically clean building block, the raw gradient it produces requires external post-processing to work inside a language model.**

These naive implementations are slow. However, the wind_rosa algorithm computes these functions efficiently with linear scaling in terms of sequence length.

See [test.py](test.py) for example usage.
* To load the CUDA kernel, call `load_wind_rosa(T=sequence_length, C=bits_per_channel, K=truncation_length)`.
* To provide an autograd-compatible interface (since PyTorch requires floating-point tensors for gradients), wind_rosa(q, k, v) wraps the kernel as follows (see [wind_rosa.py](wind_rosa.py) for details): 
  * It takes float tensors `q,k` and `v` of shape `[B,T,H*C]`.
  * Converts `q,k` and `v` to bitmasks indicating positive elements.
  * Packs the bitmasks into uint8 tensors of shape `[B,T,H]`.
  * Transposes to `[B,H,T]`.
  * Allocates scratch space (this is kept in a global buffer for reuse).
  * Runs these through the cuda kernel `torch.ops.wind_rosa.forward`.
  * Unpacks the result into a `[B,T,H*C]` float tensor.
  * Returns this float tensor with grad_fn attached.
