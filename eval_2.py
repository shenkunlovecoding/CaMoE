"""
CaMoE 可视化评测脚本 (增强版 v2)
功能：
1. 生成高亮故事
2. 修复 CUDA Padding 问题
3. 每一层的详细激活率统计
4. [新增] 1/e 常数检验
5. [新增] 详细文本分析报告
6. [新增] Token 难度分析
"""

import torch
import torch.nn.functional as F
import math
import os
from termcolor import colored
from collections import defaultdict
from camoe import CaMoE_System
from config import CONFIG_01B, CONFIG_04B
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# ================= 常数 =================
E_CONSTANT = 1 / math.e  # 0.36787944...
PHI_CONSTANT = 1 / 1.618  # 黄金比例倒数
SQRT5_2 = math.sqrt(5/2) - 1  # 另一个可能的常数

KNOWN_CONSTANTS = {
    "1/e": E_CONSTANT,
    "1/3": 1/3,
    "1/π": 1/math.pi,
    "1/φ (黄金比例)": PHI_CONSTANT,
    "1-1/e": 1 - E_CONSTANT,
    "1/2": 0.5,
    "2/5": 0.4,
    "1/4": 0.25,
}

# ================= 配置 =================
MODEL_PATH = "checkpoints/v10_step10000.pth"  
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

# ================= 分析工具 =================

def analyze_information_constants(trans_percent):
    """分析 Trans% 与信息论常数的关系"""
    
    results = {
        "measured": trans_percent,
        "analysis": []
    }
    
    # 找最接近的常数
    deviations = {}
    for name, value in KNOWN_CONSTANTS.items():
        dev = abs(trans_percent / 100 - value)
        deviations[name] = {
            "value": value * 100,
            "deviation": dev * 100,
            "deviation_pct": dev / value * 100 if value > 0 else float('inf')
        }
    
    # 按偏差排序
    sorted_constants = sorted(deviations.items(), key=lambda x: x[1]["deviation"])
    results["closest"] = sorted_constants[0]
    results["all_comparisons"] = sorted_constants
    
    # 判断
    closest_name, closest_data = sorted_constants[0]
    if closest_data["deviation"] < 1.0:
        results["verdict"] = f"🤯 几乎完美匹配 {closest_name}!"
        results["confidence"] = "HIGH"
    elif closest_data["deviation"] < 3.0:
        results["verdict"] = f"🔥 非常接近 {closest_name}"
        results["confidence"] = "MEDIUM"
    elif closest_data["deviation"] < 5.0:
        results["verdict"] = f"👀 可能与 {closest_name} 相关"
        results["confidence"] = "LOW"
    else:
        results["verdict"] = "❓ 未匹配已知常数"
        results["confidence"] = "NONE"
    
    return results

def categorize_token(word):
    """对 Token 进行分类，用于分析路由偏好"""
    word_lower = word.lower().strip()
    
    if not word_lower:
        return "空白"
    elif word_lower in ['.', ',', '!', '?', ';', ':', '"', "'", '-']:
        return "标点"
    elif word_lower in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being']:
        return "冠词/系动词"
    elif word_lower in ['he', 'she', 'it', 'they', 'we', 'i', 'you', 'his', 'her', 'their']:
        return "代词"
    elif word_lower in ['and', 'but', 'or', 'so', 'because', 'if', 'when', 'while', 'although']:
        return "连词"
    elif word_lower in ['in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of']:
        return "介词"
    elif word_lower in ['said', 'asked', 'replied', 'thought', 'wanted', 'decided', 'felt']:
        return "叙事动词"
    elif word_lower in ['very', 'really', 'so', 'quite', 'just', 'always', 'never']:
        return "副词"
    elif word_lower in ['happy', 'sad', 'big', 'small', 'good', 'bad', 'new', 'old', 'little']:
        return "常见形容词"
    elif any(c.isdigit() for c in word_lower):
        return "数字"
    elif word_lower[0].isupper() if word_lower else False:
        return "专有名词"
    else:
        return "其他词汇"

# ================= 辅助函数 =================

def sample_top_p(probs, p, temperature):
    probs = probs.pow(1.0 / temperature)
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
    
    print("\n" + "=" * 20 + " GENERATION START " + "=" * 20)
    print(f"Prompt: {prompt}\n")
    print("-" * 50)
    print(prompt, end="", flush=True)
    
    # 统计数据初始化
    total_generated = 0
    global_trans_count = 0
    global_rwkv_count = 0
    
    # 每层的 Transformer 计数
    layer_trans_counts = {i: 0 for i in range(config['n_layer'])}
    
    # [新增] Token 级别的详细记录
    token_records = []
    
    # [新增] Token 类别统计
    category_stats = defaultdict(lambda: {"trans": 0, "rwkv": 0, "total": 0})
    
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
            logits, info = model(x_padded, step=30000, phase="normal")
            
            # 3. 获取数据
            target_idx = T_actual - 1
            next_token_logits = logits[:, target_idx, :]
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = sample_top_p(probs, top_p, temperature)
            
            # 4. 统计每一层的 Winner
            transformer_id = config['num_rwkv_experts']
            token_uses_transformer = False
            token_layer_pattern = []
            
            for layer_idx, layer_winners in enumerate(info["winners"]):
                winner_id = layer_winners[0, target_idx].item()
                
                if winner_id == transformer_id:
                    layer_trans_counts[layer_idx] += 1
                    token_uses_transformer = True
                    token_layer_pattern.append("T")
                else:
                    token_layer_pattern.append(str(winner_id))
            
            # 全局统计
            if token_uses_transformer:
                global_trans_count += 1
                color = 'red'
            else:
                global_rwkv_count += 1
                color = 'cyan'
            
            total_generated += 1
            
            # 解码
            try:
                word = tokenizer.decode([next_token.item()])
            except:
                word = ""
            
            # [新增] 记录详细信息
            token_record = {
                "token_id": next_token.item(),
                "word": word,
                "used_trans": token_uses_transformer,
                "layer_pattern": "".join(token_layer_pattern),
                "trans_layers": sum(1 for p in token_layer_pattern if p == "T"),
            }
            token_records.append(token_record)
            
            # [新增] Token 类别统计
            category = categorize_token(word)
            category_stats[category]["total"] += 1
            if token_uses_transformer:
                category_stats[category]["trans"] += 1
            else:
                category_stats[category]["rwkv"] += 1
            
            # 打印
            print(colored(word, color), end="", flush=True)
            
            x = torch.cat([x, next_token.view(1, 1)], dim=1)
            if next_token.item() == 0:
                break
    
    print("\n" + "-" * 50)
    
    # ================= 分析报告 =================
    
    if total_generated > 0:
        trans_percent = global_trans_count / total_generated * 100
        
        print("\n" + "=" * 50)
        print("📊 GENERATION STATISTICS")
        print("=" * 50)
        
        print(f"\n📈 Global Stats: {total_generated} tokens")
        print(f"   🔵 RWKV Token: {global_rwkv_count} ({global_rwkv_count/total_generated:.1%})")
        print(f"   🔴 Trans Token: {global_trans_count} ({global_trans_count/total_generated:.1%})")
        print(f"   (注: Token 只要有一层用了 Trans 就算红色)")
        
        # ===== 层级分析 =====
        print("\n" + "-" * 50)
        print("🔍 Layer-wise Transformer Usage:")
        print("Layer | Usage % | Visualization")
        print("-" * 45)
        
        layer_percents = []
        for i in range(config['n_layer']):
            count = layer_trans_counts[i]
            pct = count / total_generated * 100
            layer_percents.append(pct)
            
            bar_len = int(pct / 5)  # 每5%一格
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            if pct > 50:
                indicator = "🔴"
            elif pct > 20:
                indicator = "🟡"
            else:
                indicator = "🟢"
            
            print(f" L{i:02d}  | {pct:5.1f}%  | {bar} {indicator}")
        
        # ===== 1/e 常数检验 =====
        print("\n" + "-" * 50)
        print("🔬 INFORMATION THEORY ANALYSIS")
        print("-" * 50)
        
        # 计算平均 Trans% (per layer)
        avg_layer_trans = sum(layer_percents) / len(layer_percents)
        
        # 全局 Trans% 分析
        global_analysis = analyze_information_constants(trans_percent)
        print(f"\n📍 Global Trans% (any layer): {trans_percent:.2f}%")
        print(f"   Closest constant: {global_analysis['closest'][0]} = {global_analysis['closest'][1]['value']:.2f}%")
        print(f"   Deviation: {global_analysis['closest'][1]['deviation']:.2f}%")
        print(f"   Verdict: {global_analysis['verdict']}")
        
        # 平均层级 Trans% 分析
        layer_analysis = analyze_information_constants(avg_layer_trans)
        print(f"\n📍 Average Layer Trans%: {avg_layer_trans:.2f}%")
        print(f"   Closest constant: {layer_analysis['closest'][0]} = {layer_analysis['closest'][1]['value']:.2f}%")
        print(f"   Deviation: {layer_analysis['closest'][1]['deviation']:.2f}%")
        print(f"   Verdict: {layer_analysis['verdict']}")
        
        # 特定层分析
        print("\n📍 Special Layers:")
        max_layer = max(range(len(layer_percents)), key=lambda i: layer_percents[i])
        min_layer = min(range(len(layer_percents)), key=lambda i: layer_percents[i])
        print(f"   Max Trans Layer: L{max_layer:02d} ({layer_percents[max_layer]:.1f}%)")
        print(f"   Min Trans Layer: L{min_layer:02d} ({layer_percents[min_layer]:.1f}%)")
        
        # ===== Token 类别分析 =====
        print("\n" + "-" * 50)
        print("📝 TOKEN CATEGORY ANALYSIS")
        print("-" * 50)
        print(f"{'Category':<20} | {'Total':>5} | {'Trans%':>7} | Preference")
        print("-" * 55)
        
        sorted_categories = sorted(
            category_stats.items(), 
            key=lambda x: x[1]["trans"] / x[1]["total"] if x[1]["total"] > 0 else 0,
            reverse=True
        )
        
        for category, stats in sorted_categories:
            if stats["total"] > 0:
                cat_trans_pct = stats["trans"] / stats["total"] * 100
                if cat_trans_pct > 60:
                    pref = "🔴 Trans偏好"
                elif cat_trans_pct > 40:
                    pref = "🟡 均衡"
                else:
                    pref = "🔵 RWKV偏好"
                print(f"{category:<20} | {stats['total']:>5} | {cat_trans_pct:>6.1f}% | {pref}")
        
        # ===== 层级模式分析 =====
        print("\n" + "-" * 50)
        print("🧬 LAYER PATTERN ANALYSIS")
        print("-" * 50)
        
        # 统计常见的层级模式
        pattern_counts = defaultdict(int)
        for record in token_records:
            pattern_counts[record["layer_pattern"]] += 1
        
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"Top 10 Layer Patterns (0-{config['num_rwkv_experts']-1}=RWKV, T=Trans):")
        for pattern, count in sorted_patterns:
            pct = count / total_generated * 100
            print(f"   {pattern}: {count} ({pct:.1f}%)")
        
        # ===== 导出原始数据 =====
        print("\n" + "-" * 50)
        print("📋 RAW DATA FOR ANALYSIS (Copy-Paste Friendly)")
        print("-" * 50)
        
        print("\n# Layer Trans Percentages")
        print(f"layer_trans_pct = {[round(p, 2) for p in layer_percents]}")
        
        print("\n# Global Stats")
        print(f"total_tokens = {total_generated}")
        print(f"trans_tokens = {global_trans_count}")
        print(f"rwkv_tokens = {global_rwkv_count}")
        print(f"global_trans_pct = {trans_percent:.4f}")
        print(f"avg_layer_trans_pct = {avg_layer_trans:.4f}")
        
        print("\n# Constants Comparison")
        for name, data in global_analysis['all_comparisons']:
            print(f"# {name}: expected={data['value']:.2f}%, deviation={data['deviation']:.2f}%")
        
        print("\n# Category Stats")
        print("category_trans_rates = {")
        for cat, stats in sorted_categories:
            if stats["total"] > 0:
                rate = stats["trans"] / stats["total"]
                print(f"    '{cat}': {rate:.4f},")
        print("}")
        
        # ===== 1/e 特别检验 =====
        print("\n" + "=" * 50)
        print("🎯 1/e HYPOTHESIS TEST")
        print("=" * 50)
        
        e_val = E_CONSTANT * 100
        deviation_from_e = abs(avg_layer_trans - e_val)
        
        print(f"\n   Expected 1/e:        {e_val:.2f}%")
        print(f"   Measured avg layer:  {avg_layer_trans:.2f}%")
        print(f"   Deviation:           {deviation_from_e:.2f}%")
        
        if deviation_from_e < 1.0:
            print(f"\n   🤯 RESULT: 1/e HYPOTHESIS STRONGLY SUPPORTED!")
        elif deviation_from_e < 3.0:
            print(f"\n   🔥 RESULT: 1/e hypothesis likely valid")
        elif deviation_from_e < 5.0:
            print(f"\n   👀 RESULT: Possible correlation with 1/e")
        else:
            print(f"\n   ❓ RESULT: Does not match 1/e (but may match other constants)")
        
        print("\n" + "=" * 50)

# ================= 测试用例 =================
prompts = [
    "Once upon a time, there was a little girl named Lily.",
    "The king was very sad because he lost his crown.",
]

for p in prompts:
    generate_and_visualize(p)