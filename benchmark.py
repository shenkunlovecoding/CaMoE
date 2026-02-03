"""
CaMoE v12 Benchmark Script
适配: 2 RWKV + 2 Trans 架构
"""

import torch
import torch.nn.functional as F
import time
import os
import math
from tqdm import tqdm
from camoe import CaMoE_System
from config import CONFIG_01B, CONFIG_04B
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# ================= 配置区域 =================
DATA_PATH = "data/TinyStoriesV2-GPT4-valid.txt" 
MODEL_PATH = "checkpoints/v12_step10000.pth"
SCALE = "0.1b"
DEVICE = "cuda"
CTX_LEN = 512
BATCH_SIZE = 16
CHUNK_LEN = 16

# ===========================================

def load_data_generator(path, tokenizer, ctx_len, batch_size):
    if not os.path.exists(path):
        print(f"❌ Error: Dataset not found at {path}")
        return None, 0

    print(f"📂 Reading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("🔤 Tokenizing...")
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    print(f"📊 Total tokens in eval set: {total_tokens}")

    stride = ctx_len
    num_batches = total_tokens // (batch_size * stride)
    
    limit = num_batches * batch_size * stride
    data = torch.tensor(tokens[:limit], dtype=torch.long)
    data = data.view(num_batches, batch_size, stride)
    
    return data, num_batches

def main():
    # 1. Load Config & Model
    config = CONFIG_01B if SCALE == "0.1b" else CONFIG_04B
    config['ctx_len'] = CTX_LEN
    
    NUM_RWKV = config.get('num_rwkv_experts', 2)
    NUM_TRANS = config.get('num_trans_experts', 2)
    NUM_EXPERTS = NUM_RWKV + NUM_TRANS
    
    print(f"🏗️ Loading model from {MODEL_PATH}...")
    print(f"⚙️ Config: {NUM_RWKV} RWKV + {NUM_TRANS} Trans")
    
    model = CaMoE_System(config).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 2. Tokenizer
    tokenizer = TRIE_TOKENIZER(config['vocab_file'])
    
    # 3. Data Loader
    data_loader, num_batches = load_data_generator(DATA_PATH, tokenizer, CTX_LEN, BATCH_SIZE)
    if data_loader is None:
        return

    # 4. Stats Init
    total_nll = 0.0
    total_tokens_processed = 0
    start_time = time.time()
    
    # 专家使用统计 (每层每专家)
    layer_expert_counts = {l: {e: 0 for e in range(NUM_EXPERTS)} for l in range(config['n_layer'])}
    total_decisions = 0
    
    print(f"🚀 Start Benchmarking (Batches: {num_batches})...")
    
    # 5. Eval Loop
    with torch.no_grad():
        pbar = tqdm(data_loader, total=num_batches, desc="Benchmarking")
        for batch in pbar:
            batch = batch.to(DEVICE)
            B, T = batch.shape
            
            if T % CHUNK_LEN != 0:
                T_new = (T // CHUNK_LEN) * CHUNK_LEN
                batch = batch[:, :T_new]
            
            logits, info = model(batch, step=30000, phase="normal")
            
            # Loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, config['vocab_size']), 
                shift_labels.view(-1), 
                reduction='sum'
            )
            
            total_nll += loss.item()
            total_tokens_processed += shift_labels.numel()
            
            # 专家使用统计
            for layer_idx, winners in enumerate(info['winners']):
                for e in range(NUM_EXPERTS):
                    count = (winners == e).sum().item()
                    layer_expert_counts[layer_idx][e] += count
            
            total_decisions += winners.numel() * config['n_layer']
            
            curr_ppl = math.exp(total_nll / total_tokens_processed)
            pbar.set_postfix({'PPL': f"{curr_ppl:.3f}"})

    # 6. Final Report
    end_time = time.time()
    duration = end_time - start_time
    tps = total_tokens_processed / duration
    final_ppl = math.exp(total_nll / total_tokens_processed)
    
    def get_expert_name(e):
        return f"R{e}" if e < NUM_RWKV else f"T{e - NUM_RWKV}"
    
    print("\n" + "="*60)
    print(f"🏆 BENCHMARK RESULT (CaMoE v12 - {SCALE.upper()})")
    print("="*60)
    print(f"✅ Final PPL:        {final_ppl:.4f}")
    print(f"⏱️  Speed (TPS):      {tps:.0f} tokens/s")
    print(f"🔢 Total Tokens:     {total_tokens_processed}")
    print("-" * 60)
    
    # 专家表头
    header = "Layer |"
    for e in range(NUM_EXPERTS):
        header += f" {get_expert_name(e):>6} |"
    print(header)
    print("-" * 60)
    
    # 每层统计
    rwkv_total = 0
    trans_total = 0
    
    for l in range(config['n_layer']):
        layer_total = sum(layer_expert_counts[l].values())
        row = f" L{l:02d}  |"
        for e in range(NUM_EXPERTS):
            pct = layer_expert_counts[l][e] / layer_total * 100 if layer_total > 0 else 0
            row += f" {pct:5.1f}% |"
            
            if e < NUM_RWKV:
                rwkv_total += layer_expert_counts[l][e]
            else:
                trans_total += layer_expert_counts[l][e]
        print(row)
    
    print("-" * 60)
    
    # 汇总
    grand_total = rwkv_total + trans_total
    print(f"📊 RWKV Total: {rwkv_total/grand_total*100:.1f}%")
    print(f"📊 Trans Total: {trans_total/grand_total*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()