"""
CaMoE 模型适配 lm-evaluation-harness
用法:
    python -c "import lm_eval; results = lm_eval.simple_evaluate(model='camoe', model_args='pretrained=checkpoints/v12_final.pth', tasks=['hellaswag'])"
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance

# 你的项目导入
from camoe import CaMoE_System
from backbone import init_rwkv7_cuda
from config import CONFIG_BABYLM

try:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
except ImportError:
    TRIE_TOKENIZER = None


@register_model("camoe")
class CaMoELM(LM):
    """
    CaMoE 模型的 lm-evaluation-harness 适配器
    """
    
    def __init__(
        self,
        pretrained: str = None,
        vocab_file: str = None,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 1024,
        dtype: str = "bfloat16",
        **kwargs,  # 吸收额外参数
    ):
        super().__init__()
        
        self.config = CONFIG_BABYLM.copy()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.CHUNK_LEN = 16
        init_rwkv7_cuda()
        
        # ============ 加载模型 ============
        print(f"🏗️ Building CaMoE model...")
        self.model = CaMoE_System(self.config)
        
        if pretrained and os.path.exists(pretrained):
            print(f"📦 Loading weights from {pretrained}...")
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print("✅ Weights loaded!")
        else:
            print(f"⚠️ No pretrained weights found at {pretrained}, using random init")
        
        self.model.to(self._device)
        self.model.eval()
        
        # ============ 加载 Tokenizer ============
        vocab_file = vocab_file or self.config.get('vocab_file', 'tokenizer/rwkv_vocab_v20230424.txt')
        if TRIE_TOKENIZER and os.path.exists(vocab_file):
            self.tokenizer = TRIE_TOKENIZER(vocab_file)
            print(f"✅ Loaded TRIE_TOKENIZER from {vocab_file}")
        else:
            raise RuntimeError(f"Tokenizer not found! vocab_file={vocab_file}")
        
        self.vocab_size = self.config['vocab_size']
        
        # RWKV tokenizer 的特殊 token
        self._eot_token_id = 0  # 通常是 <s> 或 padding
        self._pad_token_id = 0
    
    # ============ 必需属性 ============
    
    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id
    
    @property
    def max_gen_toks(self) -> int:
        return 256
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def max_length(self) -> int:
        return self._max_length
    
    @max_length.setter
    def max_length(self, value: int):
        self._max_length = value
    
    # ============ Tokenizer 方法 ============
    
    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:
        ids = self.tokenizer.encode(string)
        """编码字符串为 token ids"""
        return ids
    
    def tok_decode(self, tokens: List[int]) -> str:
        """解码 token ids 为字符串"""
        return self.tokenizer.decode(tokens)
    
    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量编码，返回 (input_ids, attention_mask)"""
        
        # 编码所有字符串
        all_ids = [self.tok_encode(s) for s in strings]
        
        # 截断
        if truncation and left_truncate_len:
            all_ids = [ids[-left_truncate_len:] for ids in all_ids]
        
        # 找最大长度
        max_len = max(len(ids) for ids in all_ids)
        
        # Padding
        batch_ids = []
        attention_masks = []
        
        for ids in all_ids:
            pad_len = max_len - len(ids)
            
            if padding_side == "left":
                padded = [self._pad_token_id] * pad_len + ids
                mask = [0] * pad_len + [1] * len(ids)
            else:
                padded = ids + [self._pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            
            batch_ids.append(padded)
            attention_masks.append(mask)
        
        return (
            torch.tensor(batch_ids, dtype=torch.long, device=self._device),
            torch.tensor(attention_masks, dtype=torch.long, device=self._device),
        )
    
    # ============ 模型调用 ============
    
    def _pad_to_chunk(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        将 input_ids pad 到 CHUNK_LEN 的倍数
        返回: (padded_ids, pad_len)
        """
        B, T = input_ids.shape
        
        if T % self.CHUNK_LEN == 0:
            return input_ids, 0
        
        pad_len = self.CHUNK_LEN - (T % self.CHUNK_LEN)
        padding = torch.full(
            (B, pad_len), 
            self._pad_token_id, 
            dtype=input_ids.dtype, 
            device=input_ids.device
        )
        padded = torch.cat([input_ids, padding], dim=1)
        return padded, pad_len
    
    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        print("Running forward...")
        """Forward pass，返回 logits [B, T, V]"""
        B, T = input_ids.shape
        
        # Pad 到 CHUNK_LEN 的倍数
        padded_ids, pad_len = self._pad_to_chunk(input_ids)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=self._device.type=='cuda'):
                logits, _ = self.model(padded_ids, step=0, phase="normal")
        
        # 移除 padding 对应的 logits
        if pad_len > 0:
            logits = logits[:, :T, :]
        
        return logits
    
    # ============ 核心评估方法 ============
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        计算 P(continuation | context) 的 log-likelihood
        
        返回: [(log_prob, is_greedy), ...]
        """
        results = []
        
        # 按长度排序以优化 batching
        reorder = sorted(range(len(requests)), key=lambda i: -len(requests[i].args[0]))
        
        for i in tqdm(reorder, desc="loglikelihood", disable=len(requests) < 100):
            req = requests[i]
            context, continuation = req.args
            
            # Tokenize
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            
            # 合并并截断
            full_ids = ctx_ids + cont_ids
            if len(full_ids) > self.max_length:
                # 从左边截断，保留 continuation
                full_ids = full_ids[-self.max_length:]
                ctx_len = max(0, len(full_ids) - len(cont_ids))
            else:
                ctx_len = len(ctx_ids)
            
            cont_len = len(cont_ids)
            
            # 空 continuation 特殊处理
            if cont_len == 0:
                results.append((0.0, True))
                continue
            
            # Forward
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self._device)
            logits = self._model_call(input_ids)  # [1, T, V]
            
            # logits[t] 预测 token[t+1]
            # continuation 的第一个 token 需要 logits[ctx_len - 1] (如果 ctx_len > 0)
            # 或者 logits[0] 如果整个序列就是 continuation
            
            if ctx_len > 0:
                start_idx = ctx_len - 1
            else:
                start_idx = 0
            
            end_idx = len(full_ids) - 1
            
            cont_logits = logits[0, start_idx:end_idx]  # [cont_len, V]
            
            # 目标 tokens
            if ctx_len > 0:
                target_tokens = torch.tensor(full_ids[ctx_len:], dtype=torch.long, device=self._device)
            else:
                target_tokens = torch.tensor(full_ids[1:], dtype=torch.long, device=self._device)
            
            # 确保长度匹配
            min_len = min(cont_logits.shape[0], target_tokens.shape[0])
            cont_logits = cont_logits[:min_len]
            target_tokens = target_tokens[:min_len]
            
            # 计算 log softmax
            log_probs = F.log_softmax(cont_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
            
            # 总 log prob
            total_log_prob = token_log_probs.sum().item()
            
            # 检查是否 greedy
            greedy_tokens = cont_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == target_tokens).all().item()
            
            results.append((total_log_prob, is_greedy))
        
        # 恢复原顺序
        results = [results[reorder.index(i)] for i in range(len(requests))]
        return results
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        计算整个序列的无条件 log-likelihood (用于 perplexity)
        """
        results = []
        
        for req in tqdm(requests, desc="loglikelihood_rolling", disable=len(requests) < 100):
            (string,) = req.args
            
            tokens = self.tok_encode(string)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            if len(tokens) <= 1:
                results.append((0.0, True))
                continue
            
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self._device)
            logits = self._model_call(input_ids)  # [1, T, V]
            
            # logits[t] 预测 tokens[t+1]
            shift_logits = logits[0, :-1].float()  # [T-1, V]
            shift_targets = input_ids[0, 1:]  # [T-1]
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
            
            total_log_prob = token_log_probs.sum().item()
            
            greedy_tokens = shift_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == shift_targets).all().item()
            
            results.append((total_log_prob, is_greedy))
        
        return results
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """自回归生成"""
        results = []
        
        for req in tqdm(requests, desc="generate_until", disable=len(requests) < 100):
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            
            until = gen_kwargs.get("until", [self.tok_decode([self.eot_token_id])])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            
            ctx_ids = self.tok_encode(context)
            
            max_ctx_len = self._max_length - max_gen_toks
            if len(ctx_ids) > max_ctx_len:
                ctx_ids = ctx_ids[-max_ctx_len:]
            
            generated_ids = []
            all_ids = ctx_ids.copy()
            
            for _ in range(max_gen_toks):
                # ===== 关键：每次生成也要 pad =====
                input_ids = torch.tensor([all_ids], dtype=torch.long, device=self._device)
                padded_ids, pad_len = self._pad_to_chunk(input_ids)
                
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=self.dtype, enabled=self._device.type=='cuda'):
                        logits, _ = self.model(padded_ids, step=0, phase="normal")
                
                # 取原始序列最后一个位置的 logits
                original_len = len(all_ids)
                next_logits = logits[0, original_len - 1]  # [V]
                
                if temperature <= 0:
                    next_token = next_logits.argmax().item()
                else:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                
                generated_ids.append(next_token)
                all_ids.append(next_token)
                
                current_text = self.tok_decode(generated_ids)
                should_stop = False
                
                for stop_seq in until:
                    if stop_seq in current_text:
                        current_text = current_text.split(stop_seq)[0]
                        should_stop = True
                        break
                
                if should_stop or next_token == self.eot_token_id:
                    break
                
                if len(all_ids) >= self._max_length:
                    break
            
            generated_text = self.tok_decode(generated_ids)
            for stop_seq in until:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break
            
            results.append(generated_text)
        
        return results

# ============ 便捷函数 ============

def evaluate_camoe(
    checkpoint_path: str,
    tasks: List[str],
    num_fewshot: int = 0,
    batch_size: int = 1,
    device: str = "cuda",
    limit: Optional[int] = None,
) -> dict:
    """
    便捷函数：评估 CaMoE 模型
    
    示例:
        results = evaluate_camoe(
            checkpoint_path="checkpoints/v12_step10000.pth",
            tasks=["hellaswag", "arc_easy"],
            num_fewshot=0,
        )
    """
    import lm_eval
    
    results = lm_eval.simple_evaluate(
        model="camoe",
        model_args=f"pretrained={checkpoint_path},device={device},batch_size={batch_size}",
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )
    
    return results


# ============ 命令行入口 ============

if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Evaluate CaMoE with lm-evaluation-harness")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tasks", "-t", type=str, nargs="+", default=["hellaswag"], help="Tasks to evaluate")
    parser.add_argument("--num_fewshot", "-n", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit examples per task")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = evaluate_camoe(
        checkpoint_path=args.checkpoint,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    
    for task_name, task_results in results["results"].items():
        print(f"\n📌 {task_name}:")
        for metric, value in task_results.items():
            if not metric.endswith(",stderr"):
                stderr_key = f"{metric},stderr"
                stderr = task_results.get(stderr_key, 0)
                if isinstance(value, float):
                    print(f"   {metric}: {value:.4f} ± {stderr:.4f}")
    
    # 保存结果
    if args.output:
        import json
        from lm_eval.utils import handle_non_serializable
        
        with open(args.output, "w") as f:
            json.dump(results, f, default=handle_non_serializable, indent=2)
        print(f"\n💾 Results saved to {args.output}")