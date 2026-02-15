"""
CaMoE v18 可视化深度评测脚本 (Sherlock Edition)
适配: 6 RWKV + 2 Trans (Top-2) 架构
功能：
1. 生成带颜色高亮的故事 (人类看)
2. 自动识别 Top-2 路由状态 (双R / 混动 / 双T)
3. 统计层级 Transformer 渗透率
"""

import torch
import torch.nn.functional as F
import os
import sys
from termcolor import colored
from collections import Counter

# 确保能导入 CaMoE 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CaMoE.system import CaMoE_System
from CaMoE.config import get_config # 使用 config.py 的 getter

# 尝试导入 Rust Tokenizer，没有就用 Python 版
try:
    import pyrwkv_tokenizer
    RUST_TOKENIZER = True
except ImportError:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    RUST_TOKENIZER = False

# ================= 配置 =================
# [请确认] 模型路径
MODEL_PATH = "checkpoints/v18_0.4b/v18_step10000.pth" # 你的 Pilot 路径
SCALE = "0.4b"  # "0.4b" or "pilot" or "0.1b"
DEVICE = "cuda"

# ================= 加载逻辑 =================
config = get_config(SCALE).copy()

# 强制推理配置
config['micro_batch_size'] = 1 
config['ctx_len'] = 1024 # 推理长度
config['dropout'] = 0.0

print(f"🔄 Loading model from {MODEL_PATH}...")
print(f"⚙️ Config: {config['num_rwkv_experts']}R + {config['num_trans_experts']}T (Top-{config.get('top_k', 2)})")

model = CaMoE_System(config).to(DEVICE)

# 加载权重
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    # 兼容不同的保存格式
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Full strict load success.")
    except Exception as e:
        print(f"⚠️ Strict load failed, trying non-strict... ({str(e)[:100]}...)")
        model.load_state_dict(state_dict, strict=False)
        print("✅ Non-strict load success.")
else:
    print(f"❌ Checkpoint not found: {MODEL_PATH}")
    exit()

model.eval()

# 加载 Tokenizer
print("🔤 Loading Tokenizer...")
if RUST_TOKENIZER:
    # Rust 版不需要 vocab 文件路径，内置
    tokenizer = pyrwkv_tokenizer.RWKVTokenizer()
    print("✅ Rust Tokenizer loaded.")
elif os.path.exists(config['vocab_file']):
    tokenizer = TRIE_TOKENIZER(config['vocab_file'])
    print("✅ Python Trie Tokenizer loaded.")
else:
    print("❌ Tokenizer vocab file not found.")
    exit()

# ================= 辅助函数 =================
def sample_top_p(probs: torch.Tensor, p: float, temperature: float) -> torch.Tensor:
    r"""sample_top_p(probs, p, temperature) -> Tensor

    对概率分布执行 top-p 采样。

    Args:
      probs (Tensor): 形状 ``[B, V]`` 的概率分布。
      p (float): nucleus 截断阈值。
      temperature (float): 采样温度；``0`` 表示贪心。

    Returns:
      Tensor: 形状 ``[B, 1]`` 的采样 token id。
    """
    if temperature == 0:
        return torch.argmax(probs, dim=-1).unsqueeze(0)
    
    probs = probs.pow(1.0/temperature)
    probs = probs / probs.sum()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum()
    
    return torch.multinomial(probs, 1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    context_ids: torch.Tensor,
    penalty: float = 1.2,
) -> torch.Tensor:
    r"""apply_repetition_penalty(logits, context_ids, penalty=1.2) -> Tensor

    对上下文出现过的 token 施加重复惩罚。

    Args:
      logits (Tensor): 形状 ``[B, V]``。
      context_ids (Tensor): 形状 ``[B, Seq]``。
      penalty (float, optional): 重复惩罚系数。Default: ``1.2``。

    Returns:
      Tensor: 应用惩罚后的 logits。
    """
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, context_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, context_ids, score)
    return logits


def format_prompt(user_input: str) -> str:
    r"""format_prompt(user_input) -> str

    将用户输入包装成简单对话模板。

    Args:
      user_input (str): 用户文本。

    Returns:
      str: 拼接后的 prompt。
    """
    return f"User: {user_input}\nAssistant:"


def generate_and_visualize(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> None:
    r"""generate_and_visualize(prompt, max_new_tokens=200, temperature=1.0, top_p=0.9, repetition_penalty=1.2) -> None

    生成文本并输出按路由强度着色的可视化结果。

    Args:
      prompt (str): 输入提示词。
      max_new_tokens (int, optional): 最大生成长度。Default: ``200``。
      temperature (float, optional): 采样温度。Default: ``1.0``。
      top_p (float, optional): nucleus 阈值。Default: ``0.9``。
      repetition_penalty (float, optional): 重复惩罚。Default: ``1.2``。
    """
    # Tokenize
    if RUST_TOKENIZER:
        input_ids = tokenizer.encode(prompt)
    else:
        input_ids = tokenizer.encode(prompt)
        
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    print("\n" + "="*20 + " GENERATION START " + "="*20)
    print(f"Prompt: {prompt}\n")
    print("-" * 50)
    print(prompt, end="", flush=True)
    
    # 统计数据
    total_generated = 0
    
    # 统计每一层 Transformer 的激活次数 (Top-2 中任一激活算一次)
    layer_trans_counts = {i: 0 for i in range(config['n_layer'])}
    
    # 统计全局状态：
    # 0: Pure RWKV (Blue)
    # 1: Mixed (Yellow)
    # 2: Pure Trans (Red)
    state_counts = {0: 0, 1: 0, 2: 0}
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # v18 不再强制 padding，除非使用 CUDA Kernel 优化
            # 简单起见，这里直接输入
            curr_x = x[:, -config['ctx_len']:]
            
            # Forward
            # step=30000 确保 Eureka 关闭，完全看 Router
            # phase="normal" 开启 Market
            # 开启 AMP 以匹配训练时的精度 (BF16)
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits, info = model(curr_x, step=30000, phase="normal") 
            
            # Sampling
            target_idx = curr_x.shape[1] - 1
            next_token_logits = logits[:, target_idx, :].clone()
            # 重复惩罚：对已出现在 x 中的 token 降权
            apply_repetition_penalty(next_token_logits, x, penalty=repetition_penalty)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = sample_top_p(probs, top_p, temperature)
            
            # === Top-2 路由分析 ===
            rwkv_boundary = config['num_rwkv_experts'] # e.g. 6
            
            # 当前 Token 在所有层的 Transformer 激活数
            token_trans_intensity = 0 
            
            for layer_idx, layer_winners in enumerate(info["winners"]):
                # layer_winners: [B, T, 2] -> 取当前位置 [2]
                winners = layer_winners[0, target_idx] # tensor([idx1, idx2])
                
                # 检查 Top-2 中有几个是 Transformer
                # ID >= rwkv_boundary (6) 的是 Trans
                is_trans = (winners >= rwkv_boundary).long().sum().item()
                
                if is_trans > 0:
                    layer_trans_counts[layer_idx] += 1
                    token_trans_intensity += is_trans # 这一层贡献了 1 或 2 个 Trans 强度
            
            # === 颜色逻辑 ===
            # 总共有 16 层，每层最多 2 个 Trans，满分 32 分
            # 我们根据强度定色
            
            if token_trans_intensity == 0:
                color = 'blue'       # 纯直觉 (全 RWKV)
                state_counts[0] += 1
            elif token_trans_intensity <= 5:
                color = 'cyan'       # 轻微思考
                state_counts[1] += 1
            elif token_trans_intensity <= 12:
                color = 'yellow'     # 混合模式
                state_counts[1] += 1
            else:
                color = 'red'        # 深度思考 (大量 Transformer 介入)
                state_counts[2] += 1
            
            total_generated += 1
            
            # Decode
            try:
                word = tokenizer.decode([next_token.item()])
            except:
                word = ""
                
            print(colored(word, color), end="", flush=True)
            
            x = torch.cat([x, next_token.view(1, 1)], dim=1)
            if next_token.item() == 0: break # EOS
    
    print("\n" + "-" * 50)
    
    if total_generated > 0:
        print(f"\n📊 Global Stats: {total_generated} tokens")
        print(f"🔵 Pure RWKV: {state_counts[0]} ({state_counts[0]/total_generated:.1%})")
        print(f"🟡 Mixed:      {state_counts[1]} ({state_counts[1]/total_generated:.1%})")
        print(f"🔴 Deep Trans: {state_counts[2]} ({state_counts[2]/total_generated:.1%})")
        
        print("\n🔍 Layer-wise Transformer Usage (Top-2 Hit Rate):")
        for i in range(config['n_layer']):
            # 这一层在 Top-2 中命中 Trans 的概率
            pct = layer_trans_counts[i] / total_generated
            bar_len = int(pct * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f" L{i:02d} | {pct:.1%} | {bar}")

# ================= 测试 =================
prompts = [
    "Once upon a time, there was a little girl named Lily.",  # Story 模式不用包
    format_prompt("The capital of France is Paris, but the capital of Japan is"),
    format_prompt("If x = 5 and y = 3, then x + y equals"),
]

if __name__ == "__main__":
    for p in prompts:
        generate_and_visualize(p,temperature=1.0,top_p=0.5,repetition_penalty=2)
