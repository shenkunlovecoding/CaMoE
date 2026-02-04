import os
import sys
import multiprocessing
from datasets import load_dataset

# 环境配置
sys.setrecursionlimit(10000)
sys.path.append(os.path.join(os.path.dirname(__file__), "tokenizer"))
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

# 配置路径
VOCAB_FILE = "tokenizer/rwkv_vocab_v20230424.txt" 
DATA_DIR = "./data/dev" # 存放那一堆 .train 文件的文件夹
SAVE_PATH = "./data/dev_processed"
CTX_LEN = 1024 

print(f"🚀 Initializing Tokenizer...")
global_tokenizer = TRIE_TOKENIZER(VOCAB_FILE)

def process_and_pack(examples):
    all_token_ids = []
    for text in examples["text"]:
        # 处理空行
        if not text or not text.strip():
            continue
        
        # 编码。每个段落后面强行加一个 EOS(0) 
        # 这样即使打包在一起，模型也能通过 0 知道这是不同段落
        ids = global_tokenizer.encode(text) + [0]
        all_token_ids.extend(ids)
    
    # 核心：将超长列表切成 CTX_LEN 的块
    # 比如 2500 个 token 会切成两个 1024，剩下 452 个丢弃（或留给下一个batch）
    # 在 BabyLM 这种碎数据上，这能提升 10 倍训练效率
    output = []
    for i in range(0, len(all_token_ids), CTX_LEN):
        chunk = all_token_ids[i : i + CTX_LEN]
        if len(chunk) == CTX_LEN:
            output.append(chunk)
    
    return {"input_ids": output}

def main():
    # [关键修改] 获取文件夹内所有 .train 文件
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".dev")]
    print(f"📂 Found {len(files)} BabyLM files.")

    # 告诉 load_dataset 这是文本格式
    dataset = load_dataset("text", data_files={"train": files}, split="train")
    
    print(f"📊 Raw data total lines: {len(dataset)}")

    print("⚙️ Tokenizing & Packing (Multiprocessing)...")
    
    n_proc = min(os.cpu_count(), 8)
    
    # 执行处理
    tokenized_dataset = dataset.map(
        process_and_pack,
        batched=True,
        batch_size=1000, 
        num_proc=n_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=False 
    )

    print(f"💾 Saving to: {SAVE_PATH}")
    tokenized_dataset.save_to_disk(SAVE_PATH)
    
    # 计算最终效率
    final_tokens = len(tokenized_dataset) * CTX_LEN
    print(f"✅ 处理完成！")
    print(f"📊 最终训练样本数: {len(tokenized_dataset)}")
    print(f"📊 有效 Token 总量: {final_tokens / 1e6:.2f} M Tokens")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()