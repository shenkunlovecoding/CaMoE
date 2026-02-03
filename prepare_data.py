import os
import sys
import multiprocessing
from datasets import load_dataset

# ==========================================
# [Trick 1] 暴力拉高递归上限，防止 dill 报错
# ==========================================
sys.setrecursionlimit(10000)

# 导入本地 Tokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), "tokenizer"))

try:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
except ImportError:
    try:
        from rwkv_tokenizer import TRIE_TOKENIZER
    except ImportError:
        print("❌ Error: 找不到 rwkv_tokenizer.py")
        sys.exit(1)

# 配置
VOCAB_FILE = "tokenizer/rwkv_vocab_v20230424.txt" 
LOCAL_TXT_FILE = "./data/TinyStoriesV2-GPT4-train.txt" # 或者是你的 Parquet 文件路径
SAVE_PATH = "./data/slimpajama_6b_processed"

# ==========================================
# [Trick 2] 把 Tokenizer 初始化放到全局作用域
# 这样 Windows 子进程会自己加载它，不需要 pickle 传输
# ==========================================
print(f"🚀 Initializing Tokenizer globally...")
if os.path.exists(VOCAB_FILE):
    global_tokenizer = TRIE_TOKENIZER(VOCAB_FILE)
else:
    print(f"⚠️ Warning: 没找到 {VOCAB_FILE}，如果是主进程会报错")
    global_tokenizer = None

def process(examples):
    # 直接使用全局变量 global_tokenizer
    all_ids = []
    for text in examples["text"]:
        # 注意：你的 txt 可能有空行，加个判断
        if not text.strip():
            continue
            
        try:
            ids = global_tokenizer.encode(text)
        except Exception:
            continue # 跳过无法编码的行
            
        if len(ids) > 1024:
            ids = ids[:1024]
        
        # 只有非空才加进去
        if len(ids) > 0:
            all_ids.append(ids)
        
    return {"input_ids": all_ids}

def main():
    if global_tokenizer is None:
        print("❌ 无法启动：词表文件丢失")
        return

    print(f"📂 Loading Data from: {LOCAL_TXT_FILE}")
    
    # 自动识别 txt 或 parquet
    if LOCAL_TXT_FILE.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files={"train": LOCAL_TXT_FILE}, split="train")
    else:
        dataset = load_dataset("text", data_files={"train": LOCAL_TXT_FILE}, split="train")
    
    print(f"📊 Raw Dataset Size: {len(dataset)} rows")

    print("⚙️ Tokenizing (Multiprocessing)...")
    
    # Windows 下 n_proc 不要开太大，启动开销大
    n_proc = min(os.cpu_count(), 8)
    
    tokenized_dataset = dataset.map(
        process,
        batched=True,
        batch_size=1000, 
        num_proc=n_proc, 
        # 自动删除旧列
        remove_columns=dataset.column_names 
    )

    print(f"💾 Saving to disk: {SAVE_PATH}")
    tokenized_dataset.save_to_disk(SAVE_PATH)
    print("✅ Done! 马上运行 python train.py 吧！")

if __name__ == "__main__":
    # Windows 多进程保护
    multiprocessing.freeze_support()
    main()