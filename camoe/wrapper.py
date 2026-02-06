import os
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Union
from tqdm import tqdm
# 手动实现 chunks，不求人
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance


# 你的项目导入
from .system import CaMoE_System
from .backbone import init_rwkv7_cuda
from .config import CONFIG_MINIPILE

try:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
except ImportError:
    TRIE_TOKENIZER = None

@register_model("camoe")
class CaMoELM(LM):
    """
    CaMoE 模型的 lm-evaluation-harness 适配器 (高性能 Batched 版)
    """
    
    def __init__(
        self,
        pretrained: str = None,
        vocab_file: str = None,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 1024,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__()
        
        # 1. 基础配置
        self.config = CONFIG_MINIPILE.copy()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = int(batch_size)
        self._max_length = int(max_length)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.CHUNK_LEN = 16 # RWKV-7 Kernel 要求
        
        # 2. 初始化 CUDA Kernel (防止 JIT 死锁)
        print("⏳ Init RWKV-7 CUDA Kernel...")
        init_rwkv7_cuda()
        
        # 3. 加载模型
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
        
        # 4. 加载 Tokenizer
        vocab_file = vocab_file or self.config.get('vocab_file', 'tokenizer/rwkv_vocab_v20230424.txt')
        if TRIE_TOKENIZER and os.path.exists(vocab_file):
            self.tokenizer = TRIE_TOKENIZER(vocab_file)
            print(f"✅ Loaded TRIE_TOKENIZER from {vocab_file}")
        else:
            raise RuntimeError(f"Tokenizer not found! vocab_file={vocab_file}")
        
        self.vocab_size = self.config['vocab_size']
        self._eot_token_id = 0
        self._pad_token_id = 0 # 用于 Batch Padding
    
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
    
    # ============ Tokenizer 辅助 ============
    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:
        ids = self.tokenizer.encode(string)
        # 修复空串报错刷屏问题
        if len(ids) == 0:
            if string and string.strip(): # 只有当字符串非空且有内容时才警告
                 print(f"⚠️ Warning: Empty encoding for string: '{string}'")
        return ids
    
    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    # ============ 模型前向辅助 ============
    def _pad_to_chunk(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """将 input_ids pad 到 16 的倍数 (RWKV Kernel 要求)"""
        B, T = input_ids.shape
        if T % self.CHUNK_LEN == 0:
            return input_ids, 0
        pad_len = self.CHUNK_LEN - (T % self.CHUNK_LEN)
        padding = torch.full((B, pad_len), self._pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, padding], dim=1), pad_len
    
    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass，返回 logits [B, T, V]"""
        B, T = input_ids.shape
        padded_ids, pad_len = self._pad_to_chunk(input_ids)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                # 调用模型
                logits, _ = self.model(padded_ids, step=0, phase="normal")
        
        # 切除 Padding 部分
        if pad_len > 0:
            logits = logits[:, :T, :]
        return logits
    
    # ============ 核心评估方法 (Batched) ============
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        
        # 1. 分 Batch
        for batch in tqdm(chunks(requests, self.batch_size), total=math.ceil(len(requests)/self.batch_size), desc="loglikelihood"):
            
            # --- CPU 准备数据 ---
            batch_input_ids = []
            ctx_lens = []
            cont_lens = []
            
            for req in batch:
                context, continuation = req.args
                ctx_ids = self.tok_encode(context)
                cont_ids = self.tok_encode(continuation)
                if not cont_ids: cont_ids = [self.eot_token_id]
                
                full_ids = ctx_ids + cont_ids
                if len(full_ids) > self.max_length:
                    full_ids = full_ids[-self.max_length:]
                    curr_ctx_len = max(0, len(full_ids) - len(cont_ids))
                else:
                    curr_ctx_len = len(ctx_ids)
                
                batch_input_ids.append(full_ids)
                ctx_lens.append(curr_ctx_len)
                cont_lens.append(len(cont_ids))
            
            # Padding
            max_len = max(len(ids) for ids in batch_input_ids)
            max_len = ((max_len + self.CHUNK_LEN - 1) // self.CHUNK_LEN) * self.CHUNK_LEN
            
            padded_batch = []
            for ids in batch_input_ids:
                pad_len = max_len - len(ids)
                padded_batch.append(ids + [self._pad_token_id] * pad_len)
            
            # --- GPU 计算 ---
            input_tensor = torch.tensor(padded_batch, dtype=torch.long, device=self._device)
            B, T = input_tensor.shape
            
            # 1. Forward 一次性算出所有 Logits
            logits = self._model_call(input_tensor) # [B, T, V]
            
            # 2. 向量化计算 Loss (移除 Python 循环!)
            # Logits 预测下一个词，所以错一位
            shift_logits = logits[:, :-1, :] # [B, T-1, V]
            shift_labels = input_tensor[:, 1:] # [B, T-1]
            
            # 计算所有位置的 Log Prob
            # [B, T-1]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # 计算 Greedy 预测正确性
            greedy_tokens = shift_logits.argmax(dim=-1)
            is_correct = (greedy_tokens == shift_labels) # [B, T-1]
            
            # 3. 创建 Mask (在 GPU 上并行筛选 continuation 部分)
            # 我们只关心从 ctx_len-1 开始的 cont_len 个 token
            mask = torch.zeros((B, T-1), dtype=torch.bool, device=self._device)
            
            # 这里虽然有个循环，但是是在构建 Mask，不涉及 GPU 同步，极快
            for i, (c_len, t_len) in enumerate(zip(ctx_lens, cont_lens)):
                start = c_len - 1 if c_len > 0 else 0
                end = start + t_len
                # 边界保护
                end = min(end, T-1)
                if start < end:
                    mask[i, start:end] = True
            
            # 4. 应用 Mask 并聚合
            # 只保留 continuation 部分的概率，其他置为 0
            masked_log_probs = token_log_probs * mask
            # 求和得到每个样本的总 log_prob
            sum_log_probs = masked_log_probs.sum(dim=1)
            
            # Greedy 判断: 必须 Mask 区域内全对
            # 逻辑: (预测对的数量 & Mask) == (Mask 的总长度)
            masked_correct = is_correct & mask
            num_correct = masked_correct.sum(dim=1)
            target_lens = mask.sum(dim=1)
            all_correct = (num_correct == target_lens)
            
            # 5. 转回 CPU (整个 Batch 只有这一次同步!)
            batch_res_probs = sum_log_probs.tolist()
            batch_res_greedy = all_correct.tolist()
            
            for p, g in zip(batch_res_probs, batch_res_greedy):
                results.append((p, g))
                
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        实现抽象方法，支持 PPL 测试。
        """
        results = []
        for batch in tqdm(chunks(requests, self.batch_size), total=math.ceil(len(requests)/self.batch_size), desc="rolling"):
            batch_input_ids = []
            
            for req in batch:
                (string,) = req.args
                tokens = self.tok_encode(string)
                if len(tokens) > self.max_length: tokens = tokens[:self.max_length]
                batch_input_ids.append(tokens)
                
            # Batch Padding logic (同上)
            max_len = max(len(x) for x in batch_input_ids)
            max_len = ((max_len + self.CHUNK_LEN - 1) // self.CHUNK_LEN) * self.CHUNK_LEN
            
            padded_batch = []
            valid_lens = []
            for ids in batch_input_ids:
                valid_lens.append(len(ids))
                pad_len = max_len - len(ids)
                padded_batch.append(ids + [self._pad_token_id] * pad_len)
                
            input_tensor = torch.tensor(padded_batch, dtype=torch.long, device=self._device)
            logits = self._model_call(input_tensor)
            
            for i, length in enumerate(valid_lens):
                if length <= 1:
                    results.append((0.0, True))
                    continue
                    
                # 整个序列的 PPL
                sample_logits = logits[i, :length-1, :].float()
                sample_targets = input_tensor[i, 1:length]
                
                log_probs = F.log_softmax(sample_logits, dim=-1)
                token_log_probs = log_probs.gather(1, sample_targets.unsqueeze(1)).squeeze(1)
                
                results.append((token_log_probs.sum().item(), (sample_logits.argmax(-1) == sample_targets).all().item()))
                
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        生成任务 (如 GSM8k)。保持 Serial 生成，因为变长生成很难 Batch。
        """
        results = []
        # 使用 tqdm 显示进度
        for req in tqdm(requests, desc="generate"):
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            
            until = gen_kwargs.get("until", ["\n"])
            if isinstance(until, str): until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            
            ctx_ids = self.tok_encode(context)
            # 截断 Context
            if len(ctx_ids) > self.max_length - max_gen_toks:
                ctx_ids = ctx_ids[-(self.max_length - max_gen_toks):]
            
            all_ids = ctx_ids.copy()
            
            for _ in range(max_gen_toks):
                input_tensor = torch.tensor([all_ids], dtype=torch.long, device=self._device)
                
                # 利用 _model_call 自动 pad
                logits = self._model_call(input_tensor)
                
                next_token_logits = logits[0, -1, :]
                next_token = next_token_logits.argmax().item()
                
                all_ids.append(next_token)
                
                # Check Stop
                current_text = self.tok_decode(all_ids[len(ctx_ids):])
                stop = False
                for term in until:
                    if term in current_text:
                        stop = True
                        current_text = current_text.split(term)[0]
                        break
                if stop: break
                
                if next_token == self.eot_token_id: break
            
            if not stop:
                current_text = self.tok_decode(all_ids[len(ctx_ids):])
                
            results.append(current_text)
            
        return results