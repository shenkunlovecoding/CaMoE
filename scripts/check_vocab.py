import torch
from datasets import load_from_disk
import pyrwkv_tokenizer

def check():
    # 1. 加载一个训练样本
    ds = load_from_disk("./data/camoe_toy_mix") # 你的数据路径
    sample_ids = ds[0]['input_ids'][:20] # 取前20个token
    
    # 2. 用 Rust Tokenizer 解码
    tokenizer = pyrwkv_tokenizer.RWKVTokenizer()
    decoded_text = tokenizer.decode(sample_ids)
    
    # 3. 再次编码
    re_encoded_ids = tokenizer.encode(decoded_text)
    
    print(f"Original IDs: {sample_ids}")
    print(f"Decoded Text: {decoded_text}")
    print(f"Re-encoded  : {re_encoded_ids}")
    
    if sample_ids == re_encoded_ids:
        print("✅ Tokenizer Consistency Check Passed!")
    else:
        print("❌ MISMATCH DETECTED!")
        # 打印差异
        for i, (a, b) in enumerate(zip(sample_ids, re_encoded_ids)):
            if a != b:
                print(f"Diff at {i}: {a} vs {b}")

if __name__ == "__main__":
    check()