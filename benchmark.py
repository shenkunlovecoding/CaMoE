"""
CaMoE v12 Benchmark Script (Folder Version)
适配: 2 RWKV + 2 Trans 架构
支持: 批量读取文件夹内所有文本文件
"""

import torch
import torch.nn.functional as F
import time
import os
import math
from pathlib import Path
from tqdm import tqdm
from camoe import CaMoE_System
from config import *
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# ================= 配置区域 =================
DATA_FOLDER = "data/dev"  # 改为文件夹路径
MODEL_PATH = "checkpoints/babylm/v12_step16000.pth"
SCALE = "0.1b"
DEVICE = "cuda"
CTX_LEN = 512
BATCH_SIZE = 16
CHUNK_LEN = 16

# 文件过滤配置
FILE_EXTENSIONS = ['.txt', '.dev', '.train', '.json']  # 支持的文件后缀
RECURSIVE = False  # 是否递归读取子文件夹
MAX_FILES = None   # 限制最大读取文件数 (None=无限制)

# ===========================================

def load_data_generator(folder_path, tokenizer, ctx_len, batch_size):
    """
    从文件夹加载所有文本文件并生成数据加载器
    支持多文件合并、自动过滤、进度显示
    """
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        return None, 0

    # 收集所有符合条件的文件
    folder = Path(folder_path)
    files = []
    
    if RECURSIVE:
        # 递归搜索
        for ext in FILE_EXTENSIONS:
            files.extend(folder.rglob(f"*{ext}"))
    else:
        # 仅当前文件夹
        for ext in FILE_EXTENSIONS:
            files.extend(folder.glob(f"*{ext}"))
    
    # 去重并排序
    files = sorted(list(set(files)))
    
    if MAX_FILES:
        files = files[:MAX_FILES]
    
    if not files:
        print(f"❌ Error: No {FILE_EXTENSIONS} files found in {folder_path}")
        return None, 0
    
    print(f"📂 Found {len(files)} files in {folder_path}")
    if RECURSIVE:
        print(f"   (Recursive mode enabled)")
    
    # 逐文件读取并 tokenize
    all_tokens = []
    file_stats = []
    
    for fpath in tqdm(files, desc="🔤 Tokenizing files", unit="file"):
        try:
            # 根据后缀选择读取方式
            suffix = fpath.suffix.lower()
            
            if suffix == '.jsonl':
                # JSON Lines 格式：每行一个json，取"text"字段
                import json
                texts = []
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if isinstance(data, dict) and 'text' in data:
                                texts.append(data['text'])
                            elif isinstance(data, str):
                                texts.append(data)
                        except json.JSONDecodeError:
                            continue
                text = '\n'.join(texts)
            elif suffix == '.json':
                # 标准 JSON：尝试读取 text 或 content 字段，否则读取整个文件
                import json
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict):
                            text = '\n'.join([item.get('text', item.get('content', str(item))) for item in data])
                        else:
                            text = '\n'.join([str(item) for item in data])
                    elif isinstance(data, dict):
                        text = data.get('text', data.get('content', str(data)))
                    else:
                        text = str(data)
            else:
                # 普通文本文件
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Tokenize
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            file_stats.append((fpath.name, len(tokens)))
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to process {fpath}: {e}")
            continue
    
    if not all_tokens:
        print("❌ Error: No valid tokens extracted from files")
        return None, 0
    
    # 打印文件统计
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total files processed: {len(file_stats)}")
    print(f"   Total tokens: {len(all_tokens):,}")
    
    # 显示文件大小分布（前5大）
    if len(file_stats) > 0:
        file_stats.sort(key=lambda x: x[1], reverse=True)
        print(f"\n📁 Top 5 largest files:")
        for fname, tok_count in file_stats[:5]:
            print(f"   - {fname}: {tok_count:,} tokens")
    
    # 构建 batch 数据
    total_tokens = len(all_tokens)
    stride = ctx_len
    num_batches = total_tokens // (batch_size * stride)
    
    if num_batches == 0:
        print(f"❌ Error: Not enough tokens ({total_tokens}) for one batch (need {batch_size * stride})")
        return None, 0
    
    # 截断到 batch 整数倍
    limit = num_batches * batch_size * stride
    data = torch.tensor(all_tokens[:limit], dtype=torch.long)
    data = data.view(num_batches, batch_size, stride)
    
    print(f"📦 Batches created: {num_batches} (batch_size={batch_size}, ctx_len={ctx_len})")
    print(f"   Actual tokens used: {limit:,} ({limit/total_tokens*100:.1f}% of total)\n")
    
    return data, num_batches

def main():
    # 1. Load Config & Model
    config = CONFIG_BABYLM if SCALE == "0.1b" else CONFIG_04B
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

    # 在 Benchmark 脚本开头加入：
    test_str = "Once upon a time"
    tokens = tokenizer.encode(test_str)
    print(f"Tokens: {tokens}")
    print(f"Vocab size in config: {config['vocab_size']}")
    print(f"Max token ID in this sample: {max(tokens)}")
    assert max(tokens) < config['vocab_size'], "Vocab size mismatch!"
    
    # 3. Data Loader (文件夹版本)
    data_loader, num_batches = load_data_generator(DATA_FOLDER, tokenizer, CTX_LEN, BATCH_SIZE)
    if data_loader is None:
        return

    # 4. Stats Init
    total_nll = 0.0
    total_tokens_processed = 0
    total_characters_processed = 0
    print("📏 Calculating character count for the evaluation set...")
    
    # 获取 data_loader 涵盖的所有 tokens
    all_eval_tokens = data_loader.view(-1).tolist()
    
    # 为了避免大内存占用，分块 decode
    chunk_size = 10000
    total_characters_processed = 0
    
    for i in tqdm(range(0, len(all_eval_tokens), chunk_size), desc="Decoding to chars", leave=False):
        chunk = all_eval_tokens[i:i+chunk_size]
        text_chunk = tokenizer.decode(chunk)
        total_characters_processed += len(text_chunk)
    
    print(f"📊 Total tokens: {len(all_eval_tokens)}")
    print(f"📊 Total characters: {total_characters_processed}")
    # 计算压缩率：平均每个 token 代表多少个字符
    char_per_token = total_characters_processed / len(all_eval_tokens)
    print(f"📊 Ratio: {char_per_token:.3f} characters per token")

    start_time = time.time()
    
    # 专家使用统计 (每层每专家)
    layer_expert_counts = {l: {e: 0 for e in range(NUM_EXPERTS)} for l in range(config['n_layer'])}
    total_decisions = 0
    
    print(f"\n🚀 Start Benchmarking (Batches: {num_batches})...")
    
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
            mask = (shift_labels != 0).float() 
            loss = F.cross_entropy(
                shift_logits.view(-1, config['vocab_size']), 
                shift_labels.view(-1), 
                reduction='none'
            )
            loss = (loss * mask.view(-1)).sum()
            total_tokens_processed += mask.sum().item()
            total_nll += loss.item()
            
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
    
    # 2. 计算 BPC (Bits Per Character) - 行业标准指标
    # 使用 log2 将 NLL 转换为 bits
    bpc = (total_nll / math.log(2)) / total_characters_processed
    
    # 3. 计算“等效 Per-Character PPL”
    # 假设如果模型是按字符预测的，它的 PPL 会是多少
    ppl_char = math.exp(total_nll / total_characters_processed)

    print("-" * 60)
    print(f"📉 BPC (Bits Per Character): {bpc:.4f}")
    print(f"📉 Equivalent Char-PPL:    {ppl_char:.4f}")
    
    def get_expert_name(e):
        return f"R{e}" if e < NUM_RWKV else f"T{e - NUM_RWKV}"
    
    print("\n" + "="*60)
    print(f"🏆 BENCHMARK RESULT (CaMoE v12 - {SCALE.upper()})")
    print(f"📁 Data Source: {DATA_FOLDER}")
    print("="*60)
    print(f"✅ Final PPL:        {final_ppl:.4f}")
    print(f"⏱️  Speed (TPS):      {tps:.0f} tokens/s")
    print(f"🔢 Total Tokens:     {total_tokens_processed:,}")
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
    print(f"⏱️  Total Time: {duration:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()