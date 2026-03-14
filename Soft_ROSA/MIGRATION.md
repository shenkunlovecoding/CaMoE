# Migration From `rosa_soft`

This document explains how to migrate code from the `rosa_soft` project to the
operators exposed by this `soft_rosa` folder.

## API Mapping

### Exact Soft DP path

Old import:

```python
from rosa_soft.rosa_soft import rosa_soft_ops
```

New import:

```python
from soft_rosa import soft_rosa_ops
```

Both use the same public tensor shape:

* `query`: `[B, H, T, D]`
* `key`: `[B, H, T, D]`
* `value`: `[B, H, T, Dv]`

Key semantic difference:

* `rosa_soft.rosa_soft_ops` uses **hard forward + soft backward proxy**
* `soft_rosa.soft_rosa_ops` uses **exact soft DP in both forward and backward**

So the migration is API-compatible, but not behavior-identical.

### Debugging / reference path

If you want an exact reference path that never uses the Triton scan:

```python
from soft_rosa import soft_rosa_serial_ops
```

This is useful for:

* correctness checks,
* CPU execution,
* matching the mathematical recurrence exactly in an easy-to-read path.

### GPU parallel path

If you want to force the Triton diagonal scan on CUDA:

```python
from soft_rosa import soft_rosa_parallel_ops
```

This is the closest counterpart to "production operator mode" in this folder.

### Hard QKV-1bit path

If your code already uses a flattened wind-style layout `[B, T, H*C]` and you
want the hard QKV-1bit operator:

```python
from soft_rosa import qkv1bit_rosa_ops
```

This operator is separate from the continuous soft-DP operators above.

## Suggested Migration Steps

1. Replace the import:

```python
from rosa_soft.rosa_soft import rosa_soft_ops
```

with:

```python
from soft_rosa import soft_rosa_ops
```

2. Keep the original `[B, H, T, D]` tensor layout. No shape rewrite is needed.

3. For the first migration pass, compare:

* old path: `rosa_soft.rosa_soft_ops`
* new path: `soft_rosa_serial_ops`

This gives the easiest debugging baseline.

4. Once the integration is stable, switch to:

* `soft_rosa_ops` for automatic backend choice
* or `soft_rosa_parallel_ops` to force the CUDA/Triton scan

5. If you were using `rosa_sufa_ops` for efficiency rather than exactness:

* there is no exact 1:1 replacement in this folder
* the closest exact replacement is `soft_rosa_parallel_ops`
* but note that SUFA is still an approximation, while this project keeps the
  exact soft suffix DP

## Operator Summary

* `soft_rosa_ops`
  * exact Soft ROSA
  * `[B, H, T, D]`
  * automatic scan backend
* `soft_rosa_serial_ops`
  * same operator
  * serial diagonal scan
* `soft_rosa_parallel_ops`
  * same operator
  * Triton parallel diagonal scan on CUDA
* `qkv1bit_rosa_ops`
  * hard QKV-1bit ROSA
  * `[B, T, H*C]`
  * backend choices: `reference`, `triton`, `cuda`
