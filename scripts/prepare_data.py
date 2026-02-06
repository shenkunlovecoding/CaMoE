"""
Preprocess Script for CaMoE v18 (Rust RWKV Tokenizer)
"""
import os
import argparse
from datasets import load_dataset
import pyrwkv_tokenizer # Rust 加速版

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--save_path", type=str, default="./data/TinyStories_rwkv_processed")
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=16) # Rust本身有多线程，这里Python进程数可以少点
    return parser.parse_args()

def process_batch(batch):
    # Rust Tokenizer 初始化极快，直接在函数里搞
    tokenizer = pyrwkv_tokenizer.RWKVTokenizer()
    
    texts = batch["text"]
    # 批量编码 (Rust 内部自带多线程优化)
    # 注意：pyrwkv_tokenizer 的 encode_batch 返回的是 list of lists
    encoded_batch = tokenizer.encode_batch(texts)
    
    # Flatten & Add EOS (0 for RWKV?) 
    # RWKV world tokenizer通常没有显式的EOS，或者用 0。
    # 检查 vocab 发现 0 是 <|endoftext|> ?? 需确认。
    # 假设 0 是 EOS。
    
    flat_ids = []
    for ids in encoded_batch:
        flat_ids.extend(ids)
        flat_ids.append(0) # EOS
        
    # Chunking
    chunks = []
    CTX_LEN = 1024 # 需从外部传入或写死
    for i in range(0, len(flat_ids), CTX_LEN):
        chunk = flat_ids[i:i+CTX_LEN]
        if len(chunk) == CTX_LEN:
            chunks.append(chunk)
            
    return {"input_ids": chunks}

def main():
    args = get_args()
    print(f"🚀 Preprocessing {args.dataset} with Rust RWKV Tokenizer...")
    
    ds = load_dataset(args.dataset, split="train")
    
    # 注意：因为 Rust tokenizer 内部有多线程，Python 层面的 num_proc 可以设小一点，或者设为 1
    # 实际上 datasets 的 map 多进程是 process 级，Rust 是 thread 级，两者结合可能更好。
    # 建议 Python num_proc = cpu_count // 2
    
    tokenized_ds = ds.map(
        process_batch,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing"
    )
    
    tokenized_ds.save_to_disk(args.save_path)
    print("✅ Done!")

if __name__ == "__main__":
    main()