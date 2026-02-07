"""
Preprocess Script for CaMoE v18 (Rust RWKV Tokenizer)
支持本地文件 (json/txt/csv) 和 HF 在线数据集
"""
import os
import argparse
from datasets import load_dataset
import pyrwkv_tokenizer 

def get_args():
    parser = argparse.ArgumentParser()
    # 支持本地路径或 HF ID
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to local file (.json/.txt) or HF Dataset ID")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=16)
    
    # 显式指定文本列名 (可选)
    parser.add_argument("--text_col", type=str, default=None)
    return parser.parse_args()

def process_batch(batch, text_col=None):
    tokenizer = pyrwkv_tokenizer.RWKVTokenizer()
    
    if text_col is None:
        keys = batch.keys()
        if "text" in keys: text_col = "text"
        elif "content" in keys: text_col = "content"
        elif "dialog" in keys: text_col = "dialog"
        else: text_col = list(keys)[0]
    
    raw_data = batch[text_col]
    
    texts = []
    for item in raw_data:
        if isinstance(item, str):
            # DailyDialog 修复: 替换 __eou__ 为换行
            clean_text = item.replace(" __eou__ ", "\n").replace("__eou__", "\n").strip()
            texts.append(clean_text)
        elif isinstance(item, list):
            texts.append("\n".join(str(x) for x in item))
        else:
            texts.append(str(item))
            
    if not texts: return {"input_ids": []}
    
    encoded_batch = tokenizer.encode_batch(texts)
    
    flat_ids = []
    for ids in encoded_batch:
        flat_ids.extend(ids)
        flat_ids.append(0) # EOS
        
    chunks = []
    CTX_LEN = 1024
    for i in range(0, len(flat_ids), CTX_LEN):
        chunk = flat_ids[i:i+CTX_LEN]
        if len(chunk) == CTX_LEN:
            chunks.append(chunk)
            
    return {"input_ids": chunks}

def main():
    args = get_args()
    print(f"🚀 Processing: {args.dataset}")
    
    # 1. 智能加载逻辑
    # 检查是否是本地文件
    if os.path.exists(args.dataset):
        ext = args.dataset.split('.')[-1]
        if ext in ['json', 'jsonl']:
            print("📂 Detected Local JSON/JSONL file")
            ds = load_dataset("json", data_files=args.dataset, split="train")
        elif ext == 'txt':
            print("📂 Detected Local TXT file")
            ds = load_dataset("text", data_files=args.dataset, split="train")
        elif os.path.isdir(args.dataset):
             print("📂 Detected Local Dataset Folder (Arrow/HF format)")
             from datasets import load_from_disk
             ds = load_from_disk(args.dataset)
        else:
            # 尝试作为 CSV
            ds = load_dataset("csv", data_files=args.dataset, split="train")
    else:
        # 假设是 HF Hub ID
        print("☁️  Loading from HF Hub...")
        # 即使这里不加 trust_remote_code，对于标准 dataset (text, json) 也没问题
        # 如果是特殊 script dataset，可能还会挂，但我们主要用 json/text
        ds = load_dataset(args.dataset, split="train", trust_remote_code=True)

    print(f"📊 Rows: {len(ds)}")
    
    # 2. Map 处理
    # 使用 lambda 传入 text_col 参数
    tokenized_ds = ds.map(
        lambda x: process_batch(x, args.text_col),
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing"
    )
    
    tokenized_ds.save_to_disk(args.save_path)
    print(f"✅ Saved to {args.save_path}")

if __name__ == "__main__":
    main()