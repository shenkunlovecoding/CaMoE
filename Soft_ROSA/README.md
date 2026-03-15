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

```text
y_i = sum_j softmax_j(gamma * ell(i, j)) * v_{j + 1}
```

这里的关键点不是“近似一个 hard forward”，而是把 suffix DP 本身直接变成可微对象。和旧 `rosa_soft` 相比，这里的 exact 路径在 forward 与 backward 上都保持 soft DP 语义。

## 公开 API

### Exact Soft ROSA

最直接的入口是 `soft_rosa_forward()`：

```python
from Soft_ROSA import soft_rosa_forward

y = soft_rosa_forward(
    query,         # [B, T, D]
    key,           # [B, T, D]
    value,         # [B, T, Dv]
    alpha=10.0,
    gamma=5.0,
    similarity="cosine_margin",
    max_lookback=8,
    scan_backend="auto",
)
```

可选相似度：

- `dot`
- `cosine`
- `cosine_margin`

如果你更习惯 `rosa_soft` 风格的 `[B, H, T, D]` 输入，可以用：

```python
from Soft_ROSA import soft_rosa_ops

y = soft_rosa_ops(
    query,         # [B, H, T, D]
    key,           # [B, H, T, D]
    value,         # [B, H, T, Dv]
    alpha=10.0,
    gamma=5.0,
    max_lookback=8,
    scan_backend="auto",
)
```

还可以显式指定：

- `soft_rosa_serial_ops()`
- `soft_rosa_parallel_ops()`

### QKV-1bit ROSA

QKV-1bit 路径假设最后一维是展平后的独立 1-bit stream：

```python
from Soft_ROSA import qkv1bit_rosa

y = qkv1bit_rosa(
    query,         # [B, T, N]
    key,           # [B, T, N]
    value,         # [B, T, N]
    K=6,
    backend="triton",
)
```

注意事项：

- `query / key / value` 需要同 shape
- `backend` 可选 `reference`、`triton`、`cuda`
- 浮点输入会按符号量化成 1-bit
- `qkv1bit_rosa()` 是 autograd 入口，`qkv1bit_forward()` 更偏参考 / 调试

如果你的上游接口是 `[B, T, H*C]`，可以直接用：

```python
from Soft_ROSA import qkv1bit_rosa_ops

y = qkv1bit_rosa_ops(query, key, value, K=6, backend="cuda")
```

## 后端说明

### diagonal scan backend

`soft_rosa_forward()` 和 `soft_rosa_ops()` 的 `scan_backend` 最常用的是这些值：

| backend | 含义 |
| :--- | :--- |
| `auto` | 自动选择；CUDA 环境优先并行路径 |
| `serial` | 串行参考实现，最稳定但最慢 |
| `cuda` | 强制走 CUDA diagonal scan |
| `triton` | 强制走 Triton diagonal scan |
| `parallel` | 强制并行 scan 路径，优先 CUDA 扩展 |

### QKV-1bit backend

| backend | 含义 |
| :--- | :--- |
| `reference` | 纯参考实现，最适合验算 |
| `triton` | Triton kernel |
| `cuda` | CUDA extension |

## 如何运行

从 `Soft_ROSA/` 目录内运行：

```powershell
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
from Soft_ROSA import soft_rosa_ops, qkv1bit_rosa_ops
```

- 从 `Soft_ROSA/` 目录内跑 demo 时，脚本会按本地模块路径导入
- 第一次调用 CUDA backend 时会触发 `torch.utils.cpp_extension.load(...)` 动态编译
- Triton 是可选依赖，没有 Triton 时相关 backend 会报错或不可用
- exact Soft ROSA 在数学上仍然是 `O(T^2)` 级别；并行 scan 只是加速递推，不改变整体问题规模

## 适用场景与限制

适合：

- 做 exact Soft ROSA 行为验证
- 对照 hard ROSA / `rosa_soft` 的语义差异
- 在较小规模实验里验证可训练性
- 研究 1-bit 独立 stream 的 specialized backward

暂时不应过度承诺：

- 它不是一个已经打磨好的通用发布包
- 它不是所有长序列场景下都最优的生产实现
- benchmark 结果强依赖你的显卡、CUDA、Triton 和编译环境

## License

本目录自带 `LICENSE`，当前为 MIT License。
