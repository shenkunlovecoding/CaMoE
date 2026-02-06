"""
CaMoE v12.0 可视化深度评测脚本 (Sherlock Edition)
适配: 2 RWKV + 2 Trans 架构
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
from CaMoE.system import CaMoE_System
from CaMoE.config import *
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# ================= 配置 =================
# [请确认] 模型路径是否正确
MODEL_PATH = "checkpoints/minipile/v16_step12000.pth" 
# 或者用最新的 step: "checkpoints/v12/v12_step10000.pth"

SCALE = "0.1b"
DEVICE = "cuda"
ctx_len = 512
CHUNK_LEN = 16  

# ================= 加载逻辑 =================
config = CONFIG_MINIPILE if SCALE == "0.1b" else CONFIG_04B

# [重要] 必须匹配 v12 训练配置！
config['num_rwkv_experts'] = 3
config['num_trans_experts'] = 1
config['micro_batch_size'] = 1 # 推理时 BS=1

print(f"🔄 Loading model from {MODEL_PATH}...")
print(f"⚙️ Config: {config['num_rwkv_experts']} RWKV + {config['num_trans_experts']} Trans Experts")

model = CaMoE_System(config).to(DEVICE)

# 尝试加载
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

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
# 如果没有 Tokenizer 文件，会报错，请确保文件存在
if os.path.exists(config['vocab_file']):
    tokenizer = TRIE_TOKENIZER(config['vocab_file'])
else:
    print("❌ Tokenizer vocab file not found.")
    exit()

# ================= 辅助函数 =================
def sample_top_p(probs, p, temperature):
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

def generate_and_visualize(prompt, max_new_tokens=200, temperature=0.85, top_p=0.9):
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
            # phase="normal" 开启 Market
            logits, info = model(x_padded, step=30000, phase="normal") 
            
            # Sampling
            target_idx = T_actual - 1
            next_token_logits = logits[:, target_idx, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = sample_top_p(probs, top_p, temperature)
            
            # 路由统计
            active_layers = []
            rwkv_boundary = config['num_rwkv_experts']
            
            for layer_idx, layer_winners in enumerate(info["winners"]):
                # layer_winners: [B, T]
                # 取生成位置的胜者
                winner_id = layer_winners[0, target_idx].item()
                
                # [v12 适配] ID >= num_rwkv_experts 的都是 Trans
                if winner_id >= rwkv_boundary:
                    layer_trans_counts[layer_idx] += 1
                    active_layers.append(layer_idx)
            
            trans_layer_count = len(active_layers)
            
            # [升级版颜色逻辑]
            if trans_layer_count == 0:
                color = 'blue'       # 纯直觉流
            elif trans_layer_count <= 3:
                color = 'cyan'
                global_trans_count += 0.3       # 轻量级混合
            elif trans_layer_count <= 5:
                color = 'green'
                global_trans_count += 0.5      # v13 标准三明治 (支柱层介入)
            elif trans_layer_count <= 8:
                color = 'yellow'
                global_trans_count += 0.8     # 逻辑强化 (中间层也介入了)
            else:
                color = 'red'        # 高强度推理 (全线重兵压境)
                global_trans_count += 1
            
            total_generated += 1
            
            try:
                word = tokenizer.decode([next_token.item()])
            except:
                word = ""
                
            print(colored(word, color), end="", flush=True)
            
            # 记录到日志
            
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

        
        # 可选：打印详细日志
        # print(json.dumps(analysis_log, indent=2, ensure_ascii=False))

# ================= 测试 =================
prompts = [
    
    # ===== 3. 简单对话 (Switchboard 风格) =====
    "The three main steps to cook rice are: 1. Wash the rice; 2.",
    "The capital of France is Paris, but the capital of Japan is",
    "If x = 5 and y = 3, then x + y equals",
    "The three main steps to cook rice are: 1. Wash the rice; 2.",
    "Although the weather was very cold and the wind was blowing hard, the small bird decided to",
    "Based on the above mentioned analysis, we can conclude that"
]

for p in prompts:
    generate_and_visualize(p)