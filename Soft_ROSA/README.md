# Soft_ROSA

`Soft_ROSA/` 是 CaMoE 仓库内置的一套实验性 ROSA 算子实现，目标不是复刻旧 `rosa_soft` 的 proxy 语义，而是把“最长后缀匹配”直接写成可微的 exact soft dynamic programming 路径，并补上一组面向 1-bit 流的专用后端。

当前这份目录主要解决两件事：

- exact Soft ROSA：对 suffix-match DP 做连续化，并支持 autograd
- QKV-1bit ROSA：针对 `[B, T, N]` 独立 1-bit stream 的专用 hard operator 与反向路径

## 这份目录提供什么

- `soft_rosa.py`
  - `soft_rosa_forward()`：exact Soft ROSA 主实现，输入 `[B, T, D]`
  - `hard_rosa_reference()`：简单可读的 hard baseline
  - `diagonal_affine_scan_with_backend()`：对角线 scan，支持 `serial / cuda / triton / auto`
  - `symbols_to_embeddings()`：把离散 symbol 转成 one-hot embedding
- `ops.py`
  - `soft_rosa_ops()`：`[B, H, T, D]` 风格的 operator wrapper
  - `soft_rosa_serial_ops()`：强制串行 scan
  - `soft_rosa_parallel_ops()`：强制并行 scan
  - `qkv1bit_rosa_ops()`：`[B, T, H*C]` 风格的 QKV-1bit wrapper
- `qkv1bit.py`
  - `qkv1bit_forward()`：hard QKV-1bit forward，支持 `reference / triton / cuda`
  - `qkv1bit_rosa()`：带 autograd 的 QKV-1bit operator
  - `finite_diff_bwd_channelwise()` / `finite_diff_bwd_all_channels()`：参考反向实现
- `soft_rosa_cuda.py`
  - diagonal affine scan 的 CUDA 扩展封装
- `soft_rosa_triton.py`
  - diagonal affine scan 的 Triton 实现
- `qkv1bit_cuda.py`
  - QKV-1bit CUDA 扩展封装
- `qkv1bit_triton.py`
  - QKV-1bit Triton 实现
- `demo.py`
  - soft path 向 hard path 逼近的 toy demo
- `qkv1bit_demo.py`
  - QKV-1bit 正确性、梯度与 benchmark demo
- `benchmark_ops.py`
  - scan、operator wrapper 与跨项目对比 benchmark
- `MIGRATION.md`
  - 从 `rosa_soft` 迁移到这里的 API 对照

## 核心算法

对 query `q_i` 和 key `k_j`，先定义软匹配分数：

```text
mu(i, j) = sigmoid(alpha * sim(q_i, k_j))
```

然后在下三角上做 suffix-length 的连续化递推：

```text
ell(i, j) = mu(i, j) * (1 + ell(i - 1, j - 1))
```

最后只在因果位置 `j < i` 上做 value 聚合：

`y_i = sum_j softmax_j(gamma * ell(i, j)) * v_{j + 1}`

using only causal `j < i` positions.

This is the direct continuous relaxation of the hard suffix-length DP.

## Current implementation choices

* Similarity:
  * `dot`
  * `cosine`
  * `cosine_margin`

The demo uses `cosine_margin = 2 * cosine - 1` because it makes toy exact-match
experiments sharper: identical one-hot vectors map to `+1`, orthogonal one-hot
vectors map to `-1`.

* Scan:
  * CPU keeps the serial diagonal recurrence as the reference path.
  * CUDA now uses a Triton parallel diagonal scan with backward support.

* Complexity:
  * Current reference path is still `O(T^2)` work and `O(T^2)` memory.
  * It is intended as a correctness and training-behavior prototype.

## How to run

From this directory:

```bash
python demo.py
```

这个 demo 会做三件事：

- 观察 `alpha / gamma` 增大时 soft path 如何逼近 hard baseline
- 检查 query / key / value 梯度是否正常传播
- 跑一个很小的可训练例子，看 loss 是否能逼近 hard target

QKV-1bit demo：

```powershell
python qkv1bit_demo.py --device cuda --repeat 3
```

它会输出：

- channel independence 检查
- `reference / triton / cuda` forward parity
- 梯度 parity
- specialized backward 的 benchmark

综合 benchmark：

```powershell
python benchmark_ops.py --device cuda --repeat 3
```

它会比较：

- serial vs parallel diagonal scan
- `soft_rosa_*` operator wrapper
- `rosa_soft` 对应 operator
- `qkv1bit_rosa_ops` 的各 backend

## 在 CaMoE 中如何接入

CaMoE 主仓库不会直接把模型张量塞给 `Soft_ROSA`，而是通过 `camoe/soft_rosa_adapter.py` 做形状转换和后端调度：

- `[B, T, H*bits] <-> [B, H, T, D]`
- exact soft DP 路径：`soft_exact_dp*`
- QKV-1bit 路径：`soft_qkv_binary*` / `soft_qkv_binary_bipolar*`

如果你只是在 CaMoE 里切换 backend，通常不需要直接 import 这里的模块，只需要在 `ROSAExpert` 的 `backend` 里选：

- `soft_exact_dp`
- `soft_exact_dp_serial`
- `soft_exact_dp_cuda`
- `soft_exact_dp_triton`
- `soft_qkv_binary`
- `soft_qkv_binary_cuda`
- `soft_qkv_binary_triton`
- `soft_qkv_binary_bipolar`
- `soft_qkv_binary_bipolar_cuda`
- `soft_qkv_binary_bipolar_triton`

## 与 `rosa_soft` 的关系

这份目录不是 `rosa_soft` 的原地替换版，而是另一条实验路线：

- `rosa_soft.rosa_soft_ops`
  - 更偏 hard forward + soft backward proxy
- `Soft_ROSA.soft_rosa_ops`
  - exact soft DP forward + exact soft DP backward

迁移时最重要的区别就是语义不同，不能只看函数名相似就默认行为一致。具体 API 对照请看 `MIGRATION.md`。

## 构建与运行注意事项

- 这个目录当前以 vendored package 的方式被主仓库直接引用，没有单独的 `setup.py` 或 `pyproject.toml`
- 从仓库根目录使用时，推荐直接：

```python
from soft_rosa import qkv1bit_rosa

y = qkv1bit_rosa(q, k, v, K=6, backend="triton")
```

where `q/k/v` have shape `[B, T, H*C]` and the last dimension is treated as a
flat set of independent 1-bit streams. Float inputs are quantized by sign.

For the operator-style wrapper and cross-project benchmark:

```bash
python benchmark_ops.py --device cuda --repeat 1
```

This script prints:

1. serial vs parallel diagonal scan timing,
2. operator benchmark for:
   * `soft_rosa_serial_ops`
   * `soft_rosa_parallel_ops`
   * `soft_rosa_ops`
   * `rosa_soft_ops`
   * `rosa_sufa_ops`
   * `rosa_scan_ops`
3. QKV-1bit operator benchmark for:
   * `qkv1bit_rosa_ops(..., backend="reference")`
   * `qkv1bit_rosa_ops(..., backend="triton")`
   * `qkv1bit_rosa_ops(..., backend="cuda")`

## Current demo result

On the current toy sequence

`[0, 1, 0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 4]`

the reference implementation currently shows:

* Hard-match endpoint accuracy: `1.000` at `alpha=48`, `gamma=48`
* Matched-token accuracy: `1.000`
* Gradient sanity check: non-zero gradients on query, key, and value
* Tiny trainability demo:
  * initial loss: `0.155938`
  * final loss: `0.000000`
  * final matched-token accuracy: `1.000`

This is not a proof, but it is a good first signal that:

* the forward recurrence matches the intended behavior on a repeated-pattern toy
  case,
* the operator remains trainable with plain autograd,
* annealed temperatures can move the soft path toward the hard baseline.

For `qkv1bit_demo.py`, the current CUDA run shows:

* simultaneous-flip channel-slice error: `0.000000`
* forward parity: exact for `reference / triton / cuda`
* max gradient difference to channelwise reference:
  * `dq`: `0.000000`
  * `dk`: `0.000000`
  * `dv`: `0.000000`
* benchmark:
  * `B=1, T=16, N=8, K=6`: Triton `290.14x`, CUDA `558.01x`
  * `B=1, T=12, N=16, K=6`: Triton `331.16x`, CUDA `575.48x`
  * `B=1, T=12, N=32, K=6`: Triton `846.52x`, CUDA `1531.36x`

That is exactly the effect we want: when channels are independent, flipping all
channels at a fixed time step produces the same per-channel gradient information
as flipping them one-by-one, but with about `C` times fewer forward calls.

For `benchmark_ops.py`, the current CUDA run shows:

* serial vs parallel diagonal scan:
  * `B*H=4, T=64`: `139.36x`
  * `B*H=8, T=128`: `728.07x`
  * `B*H=8, T=256`: `3077.15x`
* operator benchmark:
  * `B=1, H=2, T=64, D=8`
    * `soft_rosa_serial_ops`: `245.87 ms`
    * `soft_rosa_parallel_ops`: `2.25 ms`
    * `soft_rosa_ops`: `1.75 ms`
    * `rosa_soft_ops`: `2.81 ms`
    * `rosa_sufa_ops`: `2.59 ms`
    * `rosa_scan_ops`: `6.78 ms`
  * `B=1, H=4, T=128, D=8`
    * `soft_rosa_serial_ops`: `804.37 ms`
    * `soft_rosa_parallel_ops`: `2.41 ms`
    * `soft_rosa_ops`: `1.44 ms`
    * `rosa_soft_ops`: `2.66 ms`
    * `rosa_sufa_ops`: `2.78 ms`
    * `rosa_scan_ops`: `68.57 ms`
* QKV-1bit operator benchmark:
  * `B=1, T=16, N=8, K=6`
    * `reference`: `685.33 ms`
    * `triton`: `28.84 ms`
    * `cuda`: `10.57 ms`
  * `B=1, T=12, N=16, K=6`
    * `reference`: `326.53 ms`
    * `triton`: `16.72 ms`
    * `cuda`: `16.07 ms`
  * `B=1, T=12, N=32, K=6`
    * `reference`: `280.12 ms`
    * `triton`: `16.65 ms`
    * `cuda`: `7.88 ms`

## What "correct" means in this folder

For this first prototype, "correct" means:

* the recurrence matches the intended soft DP definition,
* autograd flows cleanly,
* on a repeated toy sequence, higher temperatures move the soft operator toward
  the hard causal suffix-match baseline,
* a small optimization loop can reduce the gap further.

This folder does **not** yet claim:

* a production-quality parallel scan,
* exact convergence guarantees in all unmatched edge cases,
* a final training recipe for large-scale LLM runs.

## Next steps

* replace the serial scan with a real parallel scan kernel,
* add an optional null / no-match branch,
* add a 1-bit per-channel QKV mode directly matching the RWKV-8 setting,
* compare the training behavior against `rosa_soft` STE and `wind_rosa`.
