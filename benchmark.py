"""
CaMoE Benchmark Script
适用: TinyStoriesV2-GPT4-valid.txt / SlimPajama Validation
功能: 测算 PPL, TPS, 以及各层 Transformer 使用率
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
# 数据集路径 (请修改为你实际的路径)
DATA_PATH = "data/TinyStoriesV2-GPT4-valid.txt" 
MODEL_PATH = "checkpoints/v10_final.pth"
SCALE = "0.1b"
DEVICE = "cuda"
CTX_LEN = 512  # 评测长度
BATCH_SIZE = 4 # 增大Batch Size可以提高评测速度
CHUNK_LEN = 16 # RWKV-7 Kernel 约束

# ===========================================

def load_data_generator(path, tokenizer, ctx_len, batch_size):
    """
    流式数据加载器，自动对齐 CHUNK_LEN
    """
    if not os.path.exists(path):
        print(f"❌ Error: Dataset not found at {path}")
        return None

    print(f"📂 Reading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("🔤 Tokenizing...")
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    print(f"📊 Total tokens in eval set: {total_tokens}")

    # 1. 掐头去尾，保证总长度能整除 (Batch * Ctx_len)
    # 虽然 RWKV Kernel 只需要 seq_len 对齐 16，但为了 Batch 效率，我们让总数规整
    stride = ctx_len
    num_batches = total_tokens // (batch_size * stride)
    
    # 转为 Tensor
    # 我们只取能整除的部分，丢弃最后一点点尾巴
    limit = num_batches * batch_size * stride
    data = torch.tensor(tokens[:limit], dtype=torch.long)
    
    # Reshape: [Num_Batches, Batch_Size, Stride]
    data = data.view(num_batches, batch_size, stride)
    
    return data, num_batches

def main():
    # 1. Load Model
    config = CONFIG_01B if SCALE == "0.1b" else CONFIG_04B
    # 强制覆盖配置以匹配训练设定
    config['num_rwkv_experts'] = 1
    config['ctx_len'] = CTX_LEN
    
    print(f"🏗️ Loading model from {MODEL_PATH}...")
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
    if data_loader is None: return

    # 4. Stats Init
    total_nll = 0.0 # Negative Log Likelihood
    total_tokens_processed = 0
    start_time = time.time()
    
    # Transformer Usage Stats
    layer_trans_counts = {i: 0 for i in range(config['n_layer'])}
    total_steps_logged = 0
    
    print(f"🚀 Start Benchmarking (Batches: {num_batches})...")
    
    # 5. Eval Loop
    with torch.no_grad():
        pbar = tqdm(data_loader, total=num_batches, desc="Benchmarking")
        for batch in pbar:
            batch = batch.to(DEVICE)
            # Input: [B, T]
            # Target: 我们需要预测下一个词。
            # 通常 PPL 评测是: Input: x[0...T-1], Target: x[1...T]
            # 这里为了简单，我们直接把 Input 喂进去，然后错一位计算 Loss
            
            # 确保长度是 16 的倍数 (虽然 CTX_LEN=512 肯定是，但为了保险)
            B, T = batch.shape
            if T % CHUNK_LEN != 0:
                # 裁掉多余的
                T_new = (T // CHUNK_LEN) * CHUNK_LEN
                batch = batch[:, :T_new]
            
            # Forward
            # step=30000 模拟成熟的 Market
            logits, info = model(batch, step=30000, phase="normal")
            
            # Shift Logic for Loss
            # Logits: [B, T, V] -> [B, T-1, V]
            # Targets: [B, T]   -> [B, T-1] (shifting right)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            # Calc Loss
            loss = F.cross_entropy(
                shift_logits.view(-1, config['vocab_size']), 
                shift_labels.view(-1), 
                reduction='sum'
            )
            
            # Accumulate
            # loss 是 sum，所以直接加
            total_nll += loss.item()
            total_tokens_processed += shift_labels.numel()
            
            # Stats: Transformer Usage
            # info['winners']: List of [B, T]
            transformer_id = config['num_rwkv_experts']
            for layer_idx, winners in enumerate(info['winners']):
                # winners: [B, T]
                # 统计有多少个 token 选了 Transformer
                usage = (winners == transformer_id).float().mean().item()
                layer_trans_counts[layer_idx] += usage
            
            total_steps_logged += 1
            
            # Update Progress Bar with current PPL
            curr_ppl = math.exp(total_nll / total_tokens_processed)
            pbar.set_postfix({'PPL': f"{curr_ppl:.3f}"})

    # 6. Final Report
    end_time = time.time()
    duration = end_time - start_time
    tps = total_tokens_processed / duration
    
    final_ppl = math.exp(total_nll / total_tokens_processed)
    
    print("\n" + "="*50)
    print(f"🏆 BENCHMARK RESULT ({SCALE.upper()})")
    print("="*50)
    print(f"✅ Final PPL:        {final_ppl:.4f}")
    print(f"⏱️  Speed (TPS):      {tps:.0f} tokens/s")
    print(f"🔢 Total Tokens:     {total_tokens_processed}")
    print("-" * 50)
    print("🧠 Layer-wise Transformer Usage (Average):")
    
    avg_total_usage = 0
    for i in range(config['n_layer']):
        avg_usage = layer_trans_counts[i] / total_steps_logged
        avg_total_usage += avg_usage
        
        # Visualization
        bar_len = int(avg_usage * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f" L{i:02d} | {avg_usage*100:5.1f}% | {bar}")
        
    print("-" * 50)
    print(f"💡 System Average Trans%: {avg_total_usage / config['n_layer'] * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()