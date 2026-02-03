"""
CaMoE 可视化深度评测脚本 (Sherlock Edition)
功能：
1. 生成带颜色高亮的故事 (人类看)
2. 生成 Token 级的详细层级路由日志 (AI 分析用)
3. 自动统计 Transformer 的“口味偏好”
"""

import torch
import torch.nn.functional as F
import os
import json
from termcolor import colored
from collections import Counter
from camoe import CaMoE_System
from config import CONFIG_01B, CONFIG_04B
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# ================= 配置 =================
MODEL_PATH = "checkpoints/v10_step10000.pth"  
SCALE = "0.1b"
DEVICE = "cuda"
ctx_len = 512
CHUNK_LEN = 16  

# ================= 加载逻辑 =================
config = CONFIG_01B if SCALE == "0.1b" else CONFIG_04B

# [重要] 这里必须和训练时意外覆盖的参数一致！
# 如果你训练时 num_rwkv_experts=3 (意味着总共4专家: 3R+1T)，这里就得填3
config['num_rwkv_experts'] = 3  
config['micro_batch_size'] = 1

print(f"🔄 Loading model from {MODEL_PATH}...")
print(f"⚙️ Config: {config['num_rwkv_experts']} RWKV Experts + 1 Linear Trans")

model = CaMoE_System(config).to(DEVICE)

# 尝试加载，容忍一些形状不匹配（如果是专家数导致的）
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

try:
    model.load_state_dict(state_dict, strict=True)
    print("✅ Full strict load success.")
except Exception as e:
    print(f"⚠️ Strict load failed, trying non-strict... ({str(e)[:100]}...)")
    model.load_state_dict(state_dict, strict=False)
    print("✅ Non-strict load success (Ignore this if generation works).")

model.eval()
tokenizer = TRIE_TOKENIZER(config['vocab_file'])

# ================= 辅助函数 =================
def sample_top_p(probs, p, temperature):
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

def analyze_token_preferences(history_log):
    """
    分析 Transformer 到底喜欢吃什么词
    """
    trans_heavy_tokens = []
    trans_light_tokens = []
    
    for item in history_log:
        token = item['token'].strip()
        if not token: continue
        
        # 统计用了 Trans 的层数
        trans_layers = len(item['trans_layers'])
        if trans_layers >= 2: # 只要有2层以上用了 Trans
            trans_heavy_tokens.append(token)
        else:
            trans_light_tokens.append(token)
            
    heavy_counts = Counter(trans_heavy_tokens).most_common(10)
    
    print("\n🧐 [AI Analysis] Transformer's Favorite Tokens (Top 10):")
    print(f"这些词最容易触发 Trans: {heavy_counts}")

def generate_and_visualize(prompt, max_new_tokens=200, temperature=1.0, top_p=0.85):
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    print("\n" + "="*20 + " GENERATION START " + "="*20)
    print(f"Prompt: {prompt}\n")
    print("-" * 50)
    print(prompt, end="", flush=True)
    
    # 统计数据
    total_generated = 0
    global_trans_count = 0
    layer_trans_counts = {i: 0 for i in range(config['n_layer'])}
    
    # AI 分析日志列表
    analysis_log = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Padding
            curr_ctx = x[:, -config['ctx_len']:]
            B, T_actual = curr_ctx.shape
            remainder = T_actual % CHUNK_LEN
            if remainder != 0:
                pad_len = CHUNK_LEN - remainder
                x_padded = F.pad(curr_ctx, (0, pad_len), value=0)
            else:
                x_padded = curr_ctx
            
            # Forward
            # step=30000 确保 Eureka 关闭，完全看 Router
            logits, info = model(x_padded, step=30000, phase="normal") 
            
            # Sampling
            target_idx = T_actual - 1
            next_token_logits = logits[:, target_idx, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = sample_top_p(probs, top_p, temperature)
            
            # 路由统计
            transformer_id = config['num_rwkv_experts'] # 最后一个 ID 是 Trans
            active_layers = []
            
            for layer_idx, layer_winners in enumerate(info["winners"]):
                # layer_winners: [B, T]
                # 注意：如果使用了 Padding，target_idx 应该是不含 padding 的索引
                # 但因为我们只取最后一个生成的，这里取 target_idx 即可
                # (如果 forward 内部做了 padding 处理，这里可能需要对齐，
                # 但根据你的代码，info返回的是对齐后的，通常取最后一个有效位)
                
                # 简单起见，我们取 info['winners'] 的对应位置
                # 如果 padding 了，info 的长度是 T_padded
                # 我们的 target_idx 是 T_actual - 1
                
                winner_id = layer_winners[0, target_idx].item()
                if winner_id == transformer_id:
                    layer_trans_counts[layer_idx] += 1
                    active_layers.append(layer_idx)
            
            # 可视化颜色
            if len(active_layers) > 0:
                global_trans_count += 1
                color = 'red'
            else:
                color = 'cyan'
            
            total_generated += 1
            
            try:
                word = tokenizer.decode([next_token.item()])
            except:
                word = ""
                
            print(colored(word, color), end="", flush=True)
            
            # 记录到日志
            analysis_log.append({
                "token": word,
                "trans_layers": active_layers
            })
            
            x = torch.cat([x, next_token.view(1, 1)], dim=1)
            if next_token.item() == 0: break
    
    print("\n" + "-" * 50)
    
    if total_generated > 0:
        print(f"\n📊 Global Stats: {total_generated} tokens")
        print(f"🔵 RWKV Token: {total_generated - global_trans_count}")
        print(f"🔴 Trans Token: {global_trans_count} ({global_trans_count/total_generated:.1%})")
        
        print("\n🔍 Layer-wise Transformer Usage:")
        for i in range(config['n_layer']):
            pct = layer_trans_counts[i] / total_generated
            bar_len = int(pct * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f" L{i:02d} | {pct:.1%} | {bar}")

        # === 这里的输出发给我 ===
        analyze_token_preferences(analysis_log)
        
        print("\n📋 Raw Token Dump (Copy this to Analysis):")
        print("[")
        for i, item in enumerate(analysis_log):
            # 只打印有 Trans 介入的，或者每隔几个打印一下，防止太长
            # 这里打印详细信息
            clean_token = repr(item['token'])
            layers = item['trans_layers']
            if len(layers) > 0:
                print(f"  {{'t': {clean_token:<10}, 'L': {layers}}},")
            else:
                pass # 纯 RWKV 的就不打印了，省空间，除非你想看上下文
        print("]")

# ================= 测试 =================
prompts = [
    "Once upon a time, there was a little girl named Lily.",
    "The king was very sad because he lost his crown.",
]

for p in prompts:
    generate_and_visualize(p)