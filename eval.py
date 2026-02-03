"""
CaMoE 可视化评测脚本 (增强版)
功能：
1. 生成高亮故事
2. 修复 CUDA Padding 问题
3. [新增] 每一层的详细激活率统计
"""

import torch
import torch.nn.functional as F
import os
from termcolor import colored
from camoe import CaMoE_System
from config import CONFIG_01B, CONFIG_04B
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# ================= 配置 =================
MODEL_PATH = "checkpoints/v10_final.pth"  
SCALE = "0.1b"
DEVICE = "cuda"
ctx_len = 512
CHUNK_LEN = 16  # RWKV-7 Kernel 约束

# ================= 加载 =================
config = CONFIG_01B if SCALE == "0.1b" else CONFIG_04B
config['num_rwkv_experts'] = 3
config['micro_batch_size'] = 1

print(f"🔄 Loading model from {MODEL_PATH}...")
model = CaMoE_System(config).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location='cpu')
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)
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

def generate_and_visualize(prompt, max_new_tokens=200, temperature=1.0, top_p=0.85):
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    print("\n" + "="*20 + " GENERATION START " + "="*20)
    print(f"Prompt: {prompt}\n")
    print("-" * 50)
    print(prompt, end="", flush=True)
    
    # 统计数据初始化
    total_generated = 0
    global_trans_count = 0
    global_rwkv_count = 0
    
    # [新增] 每层的 Transformer 计数
    layer_trans_counts = {i: 0 for i in range(config['n_layer'])}
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. Padding 处理
            curr_ctx = x[:, -config['ctx_len']:]
            B, T_actual = curr_ctx.shape
            
            remainder = T_actual % CHUNK_LEN
            if remainder != 0:
                pad_len = CHUNK_LEN - remainder
                x_padded = F.pad(curr_ctx, (0, pad_len), value=0)
            else:
                x_padded = curr_ctx
            
            # 2. Forward
            # 这里 step 传 3000 (真实步数) 还是 30000 可能会影响 Capital 逻辑
            # 但在 eval 模式下 Capital 通常是冻结的，主要看 Router 行为
            logits, info = model(x_padded, step=30000, phase="normal") 
            
            # 3. 获取数据
            target_idx = T_actual - 1
            next_token_logits = logits[:, target_idx, :]
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = sample_top_p(probs, top_p, temperature)
            
            # 4. [新增] 统计每一层的 Winner
            transformer_id = config['num_rwkv_experts']
            token_uses_transformer = False # 只要有一层用了就算这个Token是红的
            
            # info["winners"] 是一个列表，长度为 n_layer
            # 每个元素是 [B, T] 的 Tensor
            for layer_idx, layer_winners in enumerate(info["winners"]):
                winner_id = layer_winners[0, target_idx].item()
                
                if winner_id == transformer_id:
                    layer_trans_counts[layer_idx] += 1
                    token_uses_transformer = True
            
            # 全局统计
            if token_uses_transformer:
                global_trans_count += 1
                color = 'red'
            else:
                global_rwkv_count += 1
                color = 'cyan'
            
            total_generated += 1
            
            # 打印
            try:
                word = tokenizer.decode([next_token.item()])
            except:
                word = ""
            print(colored(word, color), end="", flush=True)
            
            x = torch.cat([x, next_token.view(1, 1)], dim=1)
            if next_token.item() == 0: break
    
    print("\n" + "-" * 50)
    
    if total_generated > 0:
        print(f"\n📊 Global Stats: {total_generated} tokens")
        print(f"🔵 RWKV Token: {global_rwkv_count} ({global_rwkv_count/total_generated:.1%})")
        print(f"🔴 Trans Token: {global_trans_count} ({global_trans_count/total_generated:.1%})")
        print(f"   (注: Token 只要有一层用了 Trans 就算红色)")
        
        print("\n🔍 Layer-wise Transformer Usage:")
        print("Layer | Usage % | Visualization")
        print("-" * 40)
        for i in range(config['n_layer']):
            count = layer_trans_counts[i]
            pct = count / total_generated
            
            # 简单的进度条可视化
            bar_len = int(pct * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            # 高亮显示高频层
            pct_str = f"{pct:.1%}"
            if pct > 0.5:
                pct_str = colored(pct_str, 'red')
            elif pct > 0.2:
                pct_str = colored(pct_str, 'yellow')
            else:
                pct_str = colored(pct_str, 'green')
                
            print(f" L{i:02d} | {pct_str:>7} | {bar}")

# ================= 测试用例 =================
prompts = [
    "Once upon a time, there was a little girl named Lily.",
    "The king was very sad because he lost his crown.",
]

for p in prompts:
    generate_and_visualize(p)