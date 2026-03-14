# CaMoE: Capital-Driven Mixture of Experts
*Prediction-market routing for sparse experts.*

[中文版](README.zh-CN.md) | [English](README.md)

## What Is CaMoE?
CaMoE is a sparse MoE architecture where routing is treated as a prediction market instead of a learned router with auxiliary balancing losses.

In the current implementation, every market keeps:
- a wallet for each expert (`capital`, budget only)
- a market belief vector `q`
- normalized prices `p` with `sum(p)=1`
- a supervised reward critic that predicts token-level payout for each expert

Current routing path:

```text
Traditional MoE: learned router -> logits -> top-k experts
CaMoE v23:       market prices + expert wallets + reward critic -> top-1 winner
```

## Current Version
`v23.0` is the prediction-market rewrite. The runtime path no longer uses:
- Vickrey second-price auctions
- critic pairs
- REINFORCE settlement
- four-stage `prewarm / market_warm / critic_warm / full_market`

It now uses:
- shared prediction-market routing in both markets
- supervised `RewardCritic`
- token-level `shares * (reward - price)` settlement
- two phases: `uniform_warmup` -> `full_market`

## Architecture

```text
Input
  |
  v
Embedding
  |
  v  x n_layers
+------------------------------------------------------+
| Sequence Market                                      |
| - TimeMixExpert vs ROSAExpert                        |
| - shared router semantics                            |
| - hard Top-1 route, STE soft mixing in training      |
+------------------------------------------------------+
| FFN Market                                           |
| - RWKVExpert / DeepEmbed / SlimDeepEmbed             |
| - same prediction-market router                      |
| - hard Top-1 route, STE soft mixing in training      |
+------------------------------------------------------+
| Reward Critics                                       |
| - one supervised critic per market                   |
| - predict per-token reward for each routable expert  |
+------------------------------------------------------+
  |
  v
LM Head
  |
  v
Output
```

## Market Mechanics
Per layer and per market:

```python
wallet[i]   # expert budget, stored in the existing capital buffer
q[i]        # market belief state
price[i] = (1 - liquidity_floor) * softmax(q / T) + liquidity_floor / K
```

Routing:

```python
stake[i]  = bet_fraction * wallet[i]
shares[i] = stake[i] / price[i]          # computed once at batch-start

pred_reward[t, i] = sigmoid(reward_critic(x_t)[i])
score[t, i] = shares[i] * (pred_reward[t, i] - price[i])

winner[t] = argmax(score[t])
```

Training-time exploration:
- Gaussian routing noise via `routing_noise_std`
- epsilon exploration via `exploration_epsilon`

Settlement:

```python
reward[t] = sigmoid(
    reward_scale * (market_loss_ema - token_loss[t]) / (abs(market_loss_ema) + reward_eps)
)

token_profit[t] = shares[winner[t]] * (reward[t] - price[winner[t]])
wallet_update[i] = mean(token_profit[t] for tokens won by expert i, weighted over the batch)
q[i] += price_lr * (avg_reward_i - price[i])   # only for experts that won tokens
```

Important implementation detail:
- `shares` are priced once at the start of the batch from the current wallets and prices, then reused for all tokens in that batch.
- `loss_ema` is shared per market, not per expert.
- Experts that do not win tokens in a batch do not change wallet or `q`.

## Training Schedule
Default schedule is two-phase:

```text
Step 0                     uniform_warmup_steps                end
  |-----------------------------------|------------------------->
          uniform_warmup                          full_market
```

`uniform_warmup`
- uniform mixing output
- no wallet updates
- no price updates
- no reward-critic training

`full_market`
- prediction-market routing enabled
- wallet settlement enabled
- price updates enabled
- reward critic trained every step

## Repository Map
| File | Description |
| :--- | :--- |
| `camoe/model.py` | Main `CaMoE_Model` assembly and settlement loop |
| `camoe/block.py` | Sequence and FFN market routing within a block |
| `camoe/auction.py` | `PredictionMarketRouter` |
| `camoe/capital.py` | `MarketStateManager` for wallets, prices, and shared EMA |
| `camoe/expert_critic.py` | `RewardCritic` |
| `camoe/expert_timemix.py` | TimeMix sequence expert |
| `camoe/expert_rosa.py` | Wind ROSA sequence expert |
| `camoe/expert_rwkv.py` | RWKV FFN expert |
| `camoe/expert_deepembed.py` | DeepEmbed experts |
| `train.py` | Main training entrypoint |
| `scripts/train_reverse_digits.py` | Toy-task trainer and route visualizer |
| `tests/test_prediction_market.py` | Core prediction-market unit tests |

## Vendored ROSA Backends
The repository now vendors three local ROSA-family codebases directly in-tree:

- `wind_rosa/`
  - vendored hard symbolic CUDA backend
  - local adapter: `camoe/wind_rosa_adapter.py`
- `rosa_soft/`
  - vendored proxy / SUFA / scan operators
  - local adapter: `camoe/rosa_soft_adapter.py`
- `Soft_ROSA/`
  - vendored experimental exact Soft DP and QKV-1bit kernels
  - local adapter: `camoe/soft_rosa_adapter.py`

Supported `rosa_backend` values now include:

- `wind`
- `soft`
- `sufa`
- `scan`
- `soft_exact`
- `soft_exact_serial`
- `soft_exact_cuda`
- `soft_exact_triton`
- `soft_qkv1bit`
- `soft_qkv1bit_triton`
- `soft_qkv1bit_cuda`

Practical guidance:

- `wind` remains the default hard symbolic path.
- `soft_exact` uses the adapter default selection for the Soft_ROSA diagonal scan backend.
- `soft_exact_serial` forces the exact soft-DP reference path and is mainly useful as a correctness baseline.
- `soft_exact_cuda` forces the CUDA diagonal scan kernel.
- `soft_exact_triton` forces the Triton diagonal scan kernel.
- `soft_qkv1bit*` is experimental and intended for `rosa_bits == 1`; this is the path with the extra QKV-1bit acceleration.

### Backend Benchmarking
Use the backend benchmark helper to compare the routed `ROSAExpert` implementations end to end:

```bash
python scripts/benchmark_rosa_backends.py --device cuda
```

Notes:

- On CUDA, the default benchmark set now includes `soft_exact_cuda`, `soft_exact_triton`, `soft_qkv1bit_cuda`, and `soft_qkv1bit_triton` when they are applicable.
- `soft_exact_serial` is intentionally much slower and is meant for reference comparisons.
- On Windows, Triton availability still depends on the local environment; unsupported Triton paths will be reported as `skip`.

## License Notes
- `wind_rosa/` is vendored under its included MIT license.
- `Soft_ROSA/` is vendored under its included MIT license.
- `rosa_soft/` keeps its vendored license files in-tree as well.

## Quick Start
### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python train.py \
    --scale 0.1b \
    --data /path/to/tokenized_dataset \
    --save_dir checkpoints/v23 \
    --batch_size 4 \
    --seq_len 512 \
    --steps 10000 \
    --bet_fraction 0.05 \
    --price_lr 0.02 \
    --liquidity_floor 0.02 \
    --reward_scale 5.0 \
    --exploration_epsilon 0.02
```

### Toy Smoke / Visualization
```bash
python scripts/train_reverse_digits.py \
    --model_kind camoe \
    --steps 10000 \
    --no_swanlab
```

### Generate
```bash
python eval.py \
    --checkpoint /path/to/checkpoint.pth \
    --prompt "Hello" \
    --device cuda
```

### LM Eval Harness
```bash
python lmeval.py \
    --pretrained /path/to/checkpoint.pth \
    --tasks lambada_openai \
    --device cuda
```

## Important Arguments
| Argument | Meaning |
| :--- | :--- |
| `--bet_fraction` | Fraction of wallet converted to stake each forward pass |
| `--price_lr` | How quickly market beliefs `q` react to realized reward |
| `--price_temperature` | Softmax temperature for market prices |
| `--liquidity_floor` | Prevents any expert price from collapsing to zero |
| `--reward_scale` | Sharpness of reward calibration from loss improvement |
| `--reward_eps` | Numerical stabilizer in reward computation |
| `--reward_hidden_dim` | Hidden width of the reward critic |
| `--routing_noise_std` | Gaussian noise added to routing score in training |
| `--exploration_epsilon` | Random winner override rate in training |
| `--uniform_warmup_steps` | Length of the uniform warmup phase |
| `--routing_ste` | Enable STE soft mixing during training |

## Metrics To Watch
Healthy training usually looks like:

```text
price_max stays below full monopoly
wallet_gini stays bounded
wallet_min stays above floor
realized_reward_mean tracks above price for good specialists
expected_profit_mean is not flat zero
routing_entropy does not collapse too early
exploration_rate matches the configured epsilon
main LM loss keeps falling
```

The model now logs wallet-, price-, and reward-centric diagnostics instead of old bid/capital/critic-alpha metrics.

## Testing
Core prediction-market tests:

```bash
python -m unittest tests.test_prediction_market -v
```

Static syntax check:

```bash
python -m compileall camoe train.py scripts/train_reverse_digits.py tests
```

## Notes
- Existing `capital` buffers are still named `capital`, but in `v23` they mean budget, not a direct reward multiplier.
- Old checkpoints from the Vickrey / REINFORCE era are not expected to load cleanly.
- Some legacy CLI flags may still exist in helper scripts for compatibility, but they are not part of the `v23` routing semantics.
- Inference uses strict hard Top-1 routing.
