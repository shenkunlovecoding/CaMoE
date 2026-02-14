import argparse
from datasets import load_from_disk
import pyrwkv_tokenizer

def inspect_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to inspect")
    args = parser.parse_args()
    
    print(f"🕵️ Inspecting: {args.path}")
    
    # 1. 加载数据
    try:
        ds = load_from_disk(args.path)
        if 'train' in ds: ds = ds['train']
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # 2. 加载 Tokenizer
    print("🔤 Loading Rust Tokenizer...")
    tokenizer = pyrwkv_tokenizer.RWKVTokenizer()
    
    # 3. 抽样解码
    print(f"🔍 Sampling {args.n} entries...")
    for i in range(args.n):
        print(f"\n--- Sample {i} ---")
        try:
            ids = ds[i]['input_ids']
            # 截取前 200 个 token 以防太长
            preview_ids = ids[:200]
            
            text = tokenizer.decode(preview_ids)
            print(f"[Decoded Text]:\n{text}")
            print(f"\n[Raw IDs (first 10)]: {preview_ids[:10]}")
            
        except Exception as e:
            print(f"❌ Decode failed: {e}")

if __name__ == "__main__":
    inspect_data()