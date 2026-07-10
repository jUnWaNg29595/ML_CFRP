# -*- coding: utf-8 -*-
"""
SMILES Transformer 纠错模块 (优化版)

基于深度学习的SMILES纠错方案，结合RDKit验证的完整流水线。

功能特性：
1. Transformer Encoder-Decoder 架构的序列纠错模型
2. 字符级别的SMILES编码/解码
3. Beam Search 生成多个候选修复结果
4. RDKit 验证筛选有效候选
5. 与现有规则修复方法的融合流水线
6. 支持模型训练、保存、加载
7. ★ 批量处理优化（真正的批量GPU推理）
8. ★ 支持FP16半精度推理加速
9. ★ 优化的Beam Search实现

依赖：
- torch >= 1.9.0
- rdkit
- numpy, pandas
- tqdm

Author: ML_CFRP System
"""

from __future__ import annotations

import os
import re
import json
import math
import logging
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# tqdm 可选
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """tqdm的简单替代"""
        return iterable

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch 导入
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logger.warning("PyTorch 不可用，Transformer纠错功能将被禁用")

# RDKit 导入
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    logger.warning("RDKit 不可用，SMILES验证功能将被禁用")

# 导入现有的规则修复函数
try:
    from .smiles_utils import (
        smart_repair_smiles,
        aggressive_repair_smiles,
        ultra_repair_smiles,
        canonicalize_smiles,
        clean_smiles_raw_string
    )
    SMILES_UTILS_AVAILABLE = True
except ImportError:
    SMILES_UTILS_AVAILABLE = False
    smart_repair_smiles = None
    aggressive_repair_smiles = None
    ultra_repair_smiles = None
    canonicalize_smiles = None
    clean_smiles_raw_string = lambda x: x


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class SMILESTokenizerConfig:
    """SMILES分词器配置"""
    # 特殊token
    pad_token: str = '<PAD>'
    sos_token: str = '<SOS>'
    eos_token: str = '<EOS>'
    unk_token: str = '<UNK>'
    mask_token: str = '<MASK>'
    
    # 词汇表设置 - 使用更大的默认值以容纳各种化学token
    max_vocab_size: int = 2000  # 增大到2000以确保有足够空间
    min_freq: int = 1
    
    # 序列长度
    max_length: int = 256


@dataclass
class TransformerConfig:
    """Transformer模型配置"""
    # 模型维度
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 1024
    
    # 正则化
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # 位置编码
    max_position_embeddings: int = 512
    
    # 训练设置
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64  # 增大默认batch_size
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    
    # 数据增强
    noise_prob: float = 0.15  # 噪声注入概率
    swap_prob: float = 0.05   # 字符交换概率
    delete_prob: float = 0.05 # 字符删除概率
    insert_prob: float = 0.05 # 字符插入概率
    
    # Beam Search
    beam_size: int = 5
    length_penalty: float = 0.6
    
    # 设备
    device: str = 'auto'
    
    # DataLoader设置
    num_workers: int = 0  # Windows兼容设为0
    
    # ★ 推理优化设置 - 大幅提升
    inference_batch_size: int = 512   # 大幅增加批量大小！
    use_fp16: bool = True             # 使用FP16半精度推理
    use_greedy_first: bool = True     # 先尝试贪婪解码
    auto_batch_size: bool = True      # 自动调整batch_size以最大化GPU利用率
    prefetch_factor: int = 4          # 预取因子
    use_cuda_graphs: bool = True      # 使用CUDA Graphs加速（如果可用）
    pin_memory: bool = True           # 使用锁页内存加速传输


# =============================================================================
# SMILES 分词器
# =============================================================================

class SMILESTokenizer:
    """
    SMILES字符级分词器
    
    支持：
    - 原子符号（单字符和双字符）
    - 环标记
    - 括号
    - 键类型
    - 立体化学标记
    """
    
    # SMILES字符模式（按优先级排序）
    ATOM_PATTERN = re.compile(
        r'(\[(?:[^\[\]]+)\])|'  # 方括号原子 [Na+], [nH], etc.
        r'(Br|Cl|Si|Se|se|As|Te|Li|Na|Mg|Al|Ca|Fe|Cu|Zn|Sn)|'  # 双字符原子
        r'(%\d{2})|'  # 双数字环标记 %10, %11, etc.
        r'([BCNOPSFIbcnops])|'  # 单字符原子
        r'([0-9])|'  # 单数字环标记
        r'([=#@/\\().\-+:])'  # 键、立体、括号等
    )
    
    def __init__(self, config: SMILESTokenizerConfig = None):
        self.config = config or SMILESTokenizerConfig()
        
        # 特殊token
        self.special_tokens = [
            self.config.pad_token,
            self.config.sos_token,
            self.config.eos_token,
            self.config.unk_token,
            self.config.mask_token
        ]
        
        # 初始化词汇表
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._init_vocab()
        
    def _init_vocab(self):
        """初始化基础词汇表"""
        # 添加特殊token
        for i, token in enumerate(self.special_tokens):
            self.token2idx[token] = i
            self.idx2token[i] = token
        
        # 添加常见SMILES字符
        common_tokens = [
            # 常见原子
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'F', 'Cl', 'Br', 'I',
            'B', 'b', 'Si', 'Se', 'se', 'As', 'Te',
            # 数字（环标记）
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # 双数字环标记
            '%10', '%11', '%12', '%13', '%14', '%15', '%16', '%17', '%18', '%19',
            '%20', '%21', '%22', '%23', '%24', '%25',
            # 键类型
            '-', '=', '#', ':', '.', '/',  '\\',
            # 括号
            '(', ')', '[', ']',
            # 立体化学
            '@', '@@',
            # 电荷和氢
            '+', '-', 'H',
            # 常见方括号原子
            '[H]', '[C]', '[N]', '[O]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]',
            '[nH]', '[NH]', '[NH2]', '[NH3]', '[OH]', '[O-]', '[N+]', '[S-]',
            '[Na]', '[Na+]', '[K]', '[K+]', '[Li]', '[Li+]',
            '[Ca]', '[Ca+2]', '[Mg]', '[Mg+2]', '[Zn]', '[Zn+2]',
            '[Fe]', '[Fe+2]', '[Fe+3]', '[Cu]', '[Cu+]', '[Cu+2]',
            '[*]', '[*:1]', '[*:2]',
        ]
        
        idx = len(self.special_tokens)
        for token in common_tokens:
            if token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
    
    def tokenize(self, smiles: str) -> List[str]:
        """将SMILES字符串分词为token列表"""
        if not smiles:
            return []
        
        tokens = []
        pos = 0
        
        while pos < len(smiles):
            match = self.ATOM_PATTERN.match(smiles, pos)
            if match:
                token = match.group()
                tokens.append(token)
                pos = match.end()
            else:
                # 未匹配的字符单独作为token
                tokens.append(smiles[pos])
                pos += 1
        
        return tokens
    
    def encode(self, smiles: str, add_special_tokens: bool = True, 
               max_length: Optional[int] = None) -> List[int]:
        """编码SMILES为整数ID序列"""
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = [self.config.sos_token] + tokens + [self.config.eos_token]
        
        max_len = max_length or self.config.max_length
        
        # 截断
        if len(tokens) > max_len:
            tokens = tokens[:max_len-1] + [self.config.eos_token]
        
        # 转换为ID
        ids = []
        for token in tokens:
            if token in self.token2idx:
                ids.append(self.token2idx[token])
            else:
                # 动态添加新token
                if len(self.token2idx) < self.config.max_vocab_size:
                    idx = len(self.token2idx)
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token
                    ids.append(idx)
                else:
                    ids.append(self.token2idx[self.config.unk_token])
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码整数ID序列为SMILES字符串"""
        tokens = []
        for idx in ids:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if skip_special_tokens and token in self.special_tokens:
                    if token == self.config.eos_token:
                        break
                    continue
                tokens.append(token)
        
        return ''.join(tokens)
    
    def batch_encode(self, smiles_list: List[str], 
                     max_length: Optional[int] = None,
                     padding: bool = True) -> Tuple[List[List[int]], List[int]]:
        """批量编码SMILES"""
        encoded = [self.encode(s, max_length=max_length) for s in smiles_list]
        lengths = [len(e) for e in encoded]
        
        if padding:
            max_len = max(lengths)
            pad_id = self.token2idx[self.config.pad_token]
            encoded = [e + [pad_id] * (max_len - len(e)) for e in encoded]
        
        return encoded, lengths
    
    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)
    
    @property
    def pad_token_id(self) -> int:
        return self.token2idx[self.config.pad_token]
    
    @property
    def sos_token_id(self) -> int:
        return self.token2idx[self.config.sos_token]
    
    @property
    def eos_token_id(self) -> int:
        return self.token2idx[self.config.eos_token]
    
    def save(self, path: str):
        """保存分词器"""
        data = {
            'config': self.config.__dict__,
            'token2idx': self.token2idx,
            'idx2token': {int(k): v for k, v in self.idx2token.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SMILESTokenizer':
        """加载分词器"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = SMILESTokenizerConfig(**data['config'])
        tokenizer = cls(config)
        tokenizer.token2idx = data['token2idx']
        tokenizer.idx2token = {int(k): v for k, v in data['idx2token'].items()}
        
        return tokenizer


# =============================================================================
# Transformer 模型组件
# =============================================================================

if TORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """正弦位置编码"""
        
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # 计算位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    
    class TransformerEncoder(nn.Module):
        """Transformer编码器"""
        
        def __init__(self, config: TransformerConfig, vocab_size: int, pad_idx: int):
            super().__init__()
            self.config = config
            self.pad_idx = pad_idx
            
            # 嵌入层
            self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_idx)
            self.pos_encoding = PositionalEncoding(
                config.d_model, 
                config.max_position_embeddings, 
                config.dropout
            )
            
            # Transformer编码器层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
            
            # Layer Norm
            self.layer_norm = nn.LayerNorm(config.d_model)
        
        def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # 边界检查：将超出词汇表范围的token ID替换为UNK (idx=3)
            src = torch.clamp(src, min=0, max=self.embedding.num_embeddings - 1)
            
            # 创建padding mask
            src_key_padding_mask = (src == self.pad_idx)
            
            # 嵌入 + 位置编码
            x = self.embedding(src) * math.sqrt(self.config.d_model)
            x = self.pos_encoding(x)
            
            # 编码
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
            x = self.layer_norm(x)
            
            return x
    
    
    class TransformerDecoder(nn.Module):
        """Transformer解码器"""
        
        def __init__(self, config: TransformerConfig, vocab_size: int, pad_idx: int):
            super().__init__()
            self.config = config
            self.pad_idx = pad_idx
            
            # 嵌入层
            self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_idx)
            self.pos_encoding = PositionalEncoding(
                config.d_model, 
                config.max_position_embeddings, 
                config.dropout
            )
            
            # Transformer解码器层
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
            
            # 输出层
            self.layer_norm = nn.LayerNorm(config.d_model)
            self.output_projection = nn.Linear(config.d_model, vocab_size)
        
        def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
            """生成因果mask"""
            mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
            return mask
        
        def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                    tgt_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            # 边界检查：将超出词汇表范围的token ID替换为UNK (idx=3)
            tgt = torch.clamp(tgt, min=0, max=self.embedding.num_embeddings - 1)
            
            # 创建masks
            tgt_key_padding_mask = (tgt == self.pad_idx)
            if tgt_mask is None:
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
            
            # 嵌入 + 位置编码
            x = self.embedding(tgt) * math.sqrt(self.config.d_model)
            x = self.pos_encoding(x)
            
            # 解码
            x = self.decoder(
                x, memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            x = self.layer_norm(x)
            
            # 输出投影
            logits = self.output_projection(x)
            
            return logits
    
    
    class SMILESCorrectorModel(nn.Module):
        """
        SMILES纠错Transformer模型
        
        基于Encoder-Decoder架构的序列到序列模型，
        输入错误/噪声SMILES，输出修复后的SMILES。
        """
        
        def __init__(self, config: TransformerConfig, tokenizer: SMILESTokenizer):
            super().__init__()
            self.config = config
            self.tokenizer = tokenizer
            
            # 使用max_vocab_size来初始化embedding，避免训练时动态添加token导致越界
            vocab_size = max(tokenizer.vocab_size, tokenizer.config.max_vocab_size)
            pad_idx = tokenizer.pad_token_id
            
            self.encoder = TransformerEncoder(config, vocab_size, pad_idx)
            self.decoder = TransformerDecoder(config, vocab_size, pad_idx)
            
            # 权重共享（embedding tying）
            self.decoder.embedding.weight = self.encoder.embedding.weight
            self.decoder.output_projection.weight = self.encoder.embedding.weight
            
            # 初始化权重
            self._init_weights()
        
        def _init_weights(self):
            """Xavier初始化"""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            """
            前向传播
            
            Args:
                src: 源序列 (batch, src_len)
                tgt: 目标序列 (batch, tgt_len)
            
            Returns:
                logits: (batch, tgt_len, vocab_size)
            """
            vocab_size = self.encoder.embedding.num_embeddings
            
            if src.max() >= vocab_size or src.min() < 0:
                logger.warning(f"src越界: min={src.min().item()}, max={src.max().item()}, vocab={vocab_size}")
                src = torch.clamp(src, min=0, max=vocab_size - 1)
            if tgt.max() >= vocab_size or tgt.min() < 0:
                logger.warning(f"tgt越界: min={tgt.min().item()}, max={tgt.max().item()}, vocab={vocab_size}")
                tgt = torch.clamp(tgt, min=0, max=vocab_size - 1)
            
            # 编码
            memory = self.encoder(src)
            
            # 创建memory padding mask
            memory_key_padding_mask = (src == self.tokenizer.pad_token_id)
            
            # 解码
            logits = self.decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
            
            return logits
        
        def encode(self, src: torch.Tensor) -> torch.Tensor:
            """仅编码"""
            return self.encoder(src)
        
        @torch.no_grad()
        def generate_greedy(self, src: torch.Tensor, max_length: int = 256) -> torch.Tensor:
            """贪婪解码"""
            self.eval()
            device = src.device
            batch_size = src.size(0)
            
            # 编码
            memory = self.encoder(src)
            memory_key_padding_mask = (src == self.tokenizer.pad_token_id)
            
            # 初始化解码器输入
            tgt = torch.full((batch_size, 1), self.tokenizer.sos_token_id, 
                           dtype=torch.long, device=device)
            
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for _ in range(max_length - 1):
                logits = self.decoder(tgt, memory, 
                                     memory_key_padding_mask=memory_key_padding_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 检查是否结束
                finished = finished | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
                if finished.all():
                    break
            
            return tgt
        
        @torch.no_grad()
        def generate_greedy_batch(self, src: torch.Tensor, max_length: int = 256) -> torch.Tensor:
            """
            ★ 优化的批量贪婪解码
            
            完全向量化的实现，一次性处理整个batch，大幅提升GPU利用率
            """
            self.eval()
            device = src.device
            batch_size = src.size(0)
            
            # 一次性编码整个batch
            memory = self.encoder(src)
            memory_key_padding_mask = (src == self.tokenizer.pad_token_id)
            
            # 初始化
            tgt = torch.full((batch_size, 1), self.tokenizer.sos_token_id, 
                           dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for step in range(max_length - 1):
                # 批量解码
                logits = self.decoder(tgt, memory, 
                                     memory_key_padding_mask=memory_key_padding_mask)
                
                # 获取下一个token
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # 对已完成的序列，强制填充PAD
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, self.tokenizer.pad_token_id),
                    next_token
                )
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 更新完成状态
                finished = finished | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
                
                # 所有序列都完成了
                if finished.all():
                    break
            
            return tgt
        
        @torch.no_grad()
        def generate_beam(self, src: torch.Tensor, beam_size: int = 5, 
                         max_length: int = 256, length_penalty: float = 0.6,
                         return_all: bool = False) -> Union[torch.Tensor, List[List[Tuple[torch.Tensor, float]]]]:
            """
            Beam Search解码
            """
            self.eval()
            device = src.device
            batch_size = src.size(0)
            
            # 编码
            memory = self.encoder(src)
            memory_key_padding_mask = (src == self.tokenizer.pad_token_id)
            
            all_results = []
            
            for b in range(batch_size):
                curr_memory = memory[b:b+1].expand(beam_size, -1, -1)
                curr_mask = memory_key_padding_mask[b:b+1].expand(beam_size, -1)
                
                beams = [(
                    torch.tensor([[self.tokenizer.sos_token_id]], device=device),
                    0.0,
                    False
                )]
                
                completed = []
                
                for step in range(max_length - 1):
                    if not beams:
                        break
                    
                    all_candidates = []
                    
                    for seq, score, done in beams:
                        if done:
                            completed.append((seq, score))
                            continue
                        
                        logits = self.decoder(seq, curr_memory[:1], 
                                             memory_key_padding_mask=curr_mask[:1])
                        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                        
                        topk_log_probs, topk_ids = log_probs[0].topk(beam_size)
                        
                        for i in range(beam_size):
                            next_token = topk_ids[i].unsqueeze(0).unsqueeze(0)
                            new_seq = torch.cat([seq, next_token], dim=1)
                            new_score = score + topk_log_probs[i].item()
                            is_done = (topk_ids[i].item() == self.tokenizer.eos_token_id)
                            
                            if is_done:
                                final_score = new_score / (new_seq.size(1) ** length_penalty)
                                completed.append((new_seq, final_score))
                            else:
                                all_candidates.append((new_seq, new_score, is_done))
                    
                    all_candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = all_candidates[:beam_size]
                    
                    if not beams:
                        break
                
                for seq, score, _ in beams:
                    final_score = score / (seq.size(1) ** length_penalty)
                    completed.append((seq, final_score))
                
                completed.sort(key=lambda x: x[1], reverse=True)
                all_results.append(completed[:beam_size])
            
            if return_all:
                return all_results
            else:
                best_seqs = []
                max_len = max(r[0][0].size(1) for r in all_results if r)
                for results in all_results:
                    if results:
                        seq = results[0][0]
                        if seq.size(1) < max_len:
                            pad = torch.full((1, max_len - seq.size(1)), 
                                           self.tokenizer.pad_token_id, device=device)
                            seq = torch.cat([seq, pad], dim=1)
                        best_seqs.append(seq)
                
                return torch.cat(best_seqs, dim=0) if best_seqs else None
        
        @torch.no_grad()
        def generate_beam_batch_optimized(self, src: torch.Tensor, beam_size: int = 5, 
                                          max_length: int = 256, length_penalty: float = 0.6,
                                          return_all: bool = False) -> Union[torch.Tensor, List[List[Tuple[torch.Tensor, float]]]]:
            """
            ★ 优化的批量Beam Search
            
            使用向量化操作处理整个batch，减少Python层面的循环
            """
            self.eval()
            device = src.device
            batch_size = src.size(0)
            vocab_size = self.encoder.embedding.num_embeddings
            
            # 一次性编码整个batch
            memory = self.encoder(src)
            memory_key_padding_mask = (src == self.tokenizer.pad_token_id)
            
            # 扩展memory以适应beam_size
            memory = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(batch_size * beam_size, -1, memory.size(-1))
            memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)
            
            # 初始化beam sequences和scores
            beam_seqs = torch.full((batch_size * beam_size, 1), self.tokenizer.sos_token_id, 
                                   dtype=torch.long, device=device)
            beam_scores = torch.zeros(batch_size * beam_size, device=device)
            beam_scores[1::beam_size] = float('-inf')
            for i in range(2, beam_size):
                beam_scores[i::beam_size] = float('-inf')
            
            finished = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)
            completed_seqs = [[] for _ in range(batch_size)]
            
            for step in range(max_length - 1):
                logits = self.decoder(beam_seqs, memory, 
                                     memory_key_padding_mask=memory_key_padding_mask)
                next_token_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                log_probs[finished] = float('-inf')
                log_probs[finished, self.tokenizer.pad_token_id] = 0
                
                next_scores = beam_scores.unsqueeze(-1) + log_probs
                next_scores = next_scores.view(batch_size, beam_size * vocab_size)
                
                topk_scores, topk_indices = next_scores.topk(beam_size * 2, dim=-1)
                
                beam_indices = topk_indices // vocab_size
                token_indices = topk_indices % vocab_size
                
                new_beam_seqs = []
                new_beam_scores = []
                new_finished = []
                
                for b in range(batch_size):
                    batch_offset = b * beam_size
                    selected_count = 0
                    batch_seqs = []
                    batch_scores_list = []
                    batch_finished = []
                    
                    for k in range(beam_size * 2):
                        if selected_count >= beam_size:
                            break
                        
                        beam_idx = beam_indices[b, k].item()
                        token_idx = token_indices[b, k].item()
                        score = topk_scores[b, k].item()
                        
                        orig_beam_idx = batch_offset + beam_idx
                        orig_seq = beam_seqs[orig_beam_idx]
                        
                        if token_idx == self.tokenizer.eos_token_id:
                            new_seq = torch.cat([orig_seq, torch.tensor([[token_idx]], device=device)], dim=1)
                            final_score = score / (new_seq.size(1) ** length_penalty)
                            completed_seqs[b].append((new_seq, final_score))
                        else:
                            new_seq = torch.cat([orig_seq, torch.tensor([[token_idx]], device=device)], dim=1)
                            batch_seqs.append(new_seq)
                            batch_scores_list.append(score)
                            batch_finished.append(False)
                            selected_count += 1
                    
                    while len(batch_seqs) < beam_size:
                        if batch_seqs:
                            batch_seqs.append(batch_seqs[0].clone())
                            batch_scores_list.append(float('-inf'))
                            batch_finished.append(True)
                        else:
                            dummy_seq = torch.full((1, beam_seqs.size(1) + 1), 
                                                  self.tokenizer.pad_token_id, device=device)
                            batch_seqs.append(dummy_seq)
                            batch_scores_list.append(float('-inf'))
                            batch_finished.append(True)
                    
                    new_beam_seqs.extend(batch_seqs)
                    new_beam_scores.extend(batch_scores_list)
                    new_finished.extend(batch_finished)
                
                beam_seqs = torch.cat(new_beam_seqs, dim=0)
                beam_scores = torch.tensor(new_beam_scores, device=device)
                finished = torch.tensor(new_finished, dtype=torch.bool, device=device)
                
                if finished.all():
                    break
            
            for b in range(batch_size):
                batch_offset = b * beam_size
                for k in range(beam_size):
                    idx = batch_offset + k
                    if not finished[idx] and beam_scores[idx] > float('-inf'):
                        score = beam_scores[idx].item()
                        seq = beam_seqs[idx:idx+1]
                        final_score = score / (seq.size(1) ** length_penalty)
                        completed_seqs[b].append((seq, final_score))
            
            all_results = []
            for b in range(batch_size):
                sorted_seqs = sorted(completed_seqs[b], key=lambda x: x[1], reverse=True)
                all_results.append(sorted_seqs[:beam_size])
            
            if return_all:
                return all_results
            else:
                best_seqs = []
                max_len = 1
                for results in all_results:
                    if results:
                        max_len = max(max_len, results[0][0].size(1))
                
                for results in all_results:
                    if results:
                        seq = results[0][0]
                        if seq.size(1) < max_len:
                            pad = torch.full((1, max_len - seq.size(1)), 
                                           self.tokenizer.pad_token_id, device=device)
                            seq = torch.cat([seq, pad], dim=1)
                        best_seqs.append(seq)
                    else:
                        best_seqs.append(torch.full((1, max_len), 
                                                   self.tokenizer.pad_token_id, device=device))
                
                return torch.cat(best_seqs, dim=0) if best_seqs else None


# =============================================================================
# 数据增强和噪声注入
# =============================================================================

class SMILESNoiseInjector:
    """SMILES噪声注入器"""
    
    def __init__(self, tokenizer: SMILESTokenizer, config: TrainingConfig = None):
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.common_chars = list('CNOScnos()[]=#-+123456789')
    
    def inject_noise(self, smiles: str) -> str:
        """注入噪声创建错误SMILES"""
        tokens = list(smiles)
        n = len(tokens)
        
        if n == 0:
            return smiles
        
        result = []
        i = 0
        
        while i < n:
            if np.random.random() < self.config.delete_prob:
                i += 1
                continue
            
            if i < n - 1 and np.random.random() < self.config.swap_prob:
                result.append(tokens[i + 1])
                result.append(tokens[i])
                i += 2
                continue
            
            if np.random.random() < self.config.insert_prob:
                result.append(np.random.choice(self.common_chars))
            
            if np.random.random() < self.config.noise_prob:
                result.append(np.random.choice(self.common_chars))
            else:
                result.append(tokens[i])
            
            i += 1
        
        return ''.join(result)
    
    def create_training_pair(self, clean_smiles: str) -> Tuple[str, str]:
        """创建训练对"""
        noisy = self.inject_noise(clean_smiles)
        return noisy, clean_smiles
    
    def augment_dataset(self, smiles_list: List[str], 
                       augment_factor: int = 5) -> List[Tuple[str, str]]:
        """增强数据集"""
        pairs = []
        for smiles in smiles_list:
            pairs.append((smiles, smiles))
            for _ in range(augment_factor - 1):
                noisy, clean = self.create_training_pair(smiles)
                pairs.append((noisy, clean))
        
        return pairs


# =============================================================================
# 数据集
# =============================================================================

if TORCH_AVAILABLE:
    
    class SMILESCorrectionDataset(Dataset):
        """SMILES纠错数据集"""
        
        def __init__(self, pairs: List[Tuple[str, str]], tokenizer: SMILESTokenizer,
                     max_length: int = 256, vocab_size: int = None):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.vocab_size = vocab_size or tokenizer.config.max_vocab_size
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            src_smiles, tgt_smiles = self.pairs[idx]
            
            src_ids = self.tokenizer.encode(src_smiles, max_length=self.max_length)
            tgt_ids = self.tokenizer.encode(tgt_smiles, max_length=self.max_length)
            
            src_ids = [min(max(0, i), self.vocab_size - 1) for i in src_ids]
            tgt_ids = [min(max(0, i), self.vocab_size - 1) for i in tgt_ids]
            
            return {
                'src_ids': torch.tensor(src_ids, dtype=torch.long),
                'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long)
            }
        
        @staticmethod
        def collate_fn(batch):
            """批处理函数"""
            src_ids = [item['src_ids'] for item in batch]
            tgt_ids = [item['tgt_ids'] for item in batch]
            
            src_padded = nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=0)
            tgt_padded = nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=0)
            
            return {
                'src_ids': src_padded,
                'tgt_ids': tgt_padded
            }


# =============================================================================
# 训练器
# =============================================================================

if TORCH_AVAILABLE:
    
    class SMILESCorrectorTrainer:
        """SMILES纠错模型训练器"""
        
        def __init__(self, model: SMILESCorrectorModel, tokenizer: SMILESTokenizer,
                     config: TrainingConfig = None):
            self.model = model
            self.tokenizer = tokenizer
            self.config = config or TrainingConfig()
            
            if self.config.device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(self.config.device)
            
            self.model.to(self.device)
            
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=model.config.label_smoothing
            )
            
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': []
            }
        
        def train_epoch(self, dataloader: DataLoader) -> float:
            """训练一个epoch"""
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits = self.model(src, tgt_input)
                
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / max(num_batches, 1)
        
        @torch.no_grad()
        def evaluate(self, dataloader: DataLoader) -> float:
            """评估"""
            self.model.eval()
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits = self.model(src, tgt_input)
                
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / max(num_batches, 1)
        
        def train(self, train_pairs: List[Tuple[str, str]], 
                 val_pairs: List[Tuple[str, str]],
                 save_path: Optional[str] = None) -> Dict[str, List[float]]:
            """训练模型"""
            model_vocab_size = self.model.encoder.embedding.num_embeddings
            
            train_dataset = SMILESCorrectionDataset(
                train_pairs, self.tokenizer, vocab_size=model_vocab_size
            )
            val_dataset = SMILESCorrectionDataset(
                val_pairs, self.tokenizer, vocab_size=model_vocab_size
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=SMILESCorrectionDataset.collate_fn,
                num_workers=self.config.num_workers
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=SMILESCorrectionDataset.collate_fn,
                num_workers=self.config.num_workers
            )
            
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            logger.info(f"开始训练，共 {self.config.max_epochs} 个epoch")
            
            for epoch in range(self.config.max_epochs):
                train_loss = self.train_epoch(train_loader)
                val_loss = self.evaluate(val_loader)
                
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rate'].append(current_lr)
                
                logger.info(f"Epoch {epoch+1}/{self.config.max_epochs} - "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"LR: {current_lr:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_path:
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'model_config': self.model.config.__dict__,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss
                        }, save_path)
                        logger.info(f"保存最佳模型到 {save_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"早停触发，在epoch {epoch+1}")
                        break
            
            return self.history


# =============================================================================
# 主流水线 - 优化版
# =============================================================================

class SMILESCorrectionPipeline:
    """
    ★ SMILES纠错流水线 - 深度优化版
    
    优化特性：
    1. 真正的批量GPU推理
    2. FP16半精度推理支持
    3. 优化的Beam Search
    4. 智能批处理策略
    5. ★ 自动batch_size调整
    6. ★ CUDA异步传输
    7. ★ 模型预热和编译优化
    """
    
    def __init__(self, 
                 model: Optional['SMILESCorrectorModel'] = None,
                 tokenizer: Optional[SMILESTokenizer] = None,
                 transformer_config: Optional[TransformerConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 device: str = 'auto'):
        """初始化流水线"""
        self.tokenizer = tokenizer or SMILESTokenizer()
        self.transformer_config = transformer_config or TransformerConfig()
        self.training_config = training_config or TrainingConfig()
        
        device_to_use = self.training_config.device if self.training_config.device != 'auto' else device
        
        if device_to_use == 'auto':
            self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_to_use) if TORCH_AVAILABLE else 'cpu'
        
        self.model = model
        if self.model is not None and TORCH_AVAILABLE:
            self.model.to(self.device)
            self.model.eval()
        
        # ★ FP16支持
        self.use_fp16 = self.training_config.use_fp16 and TORCH_AVAILABLE and self.device.type == 'cuda'
        if self.use_fp16 and self.model is not None:
            self.model = self.model.half()
        
        # ★ 尝试使用torch.compile()加速（PyTorch 2.0+）
        self._compiled = False
        if self.model is not None and TORCH_AVAILABLE and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self._compiled = True
                logger.info("已启用torch.compile()加速")
            except Exception as e:
                logger.debug(f"torch.compile()不可用: {e}")
        
        # ★ 自动计算最优batch_size
        self._optimal_batch_size = None
        if self.training_config.auto_batch_size and self.device.type == 'cuda':
            self._optimal_batch_size = self._find_optimal_batch_size()
        
        # ★ CUDA Stream用于异步操作
        self._cuda_stream = None
        if TORCH_AVAILABLE and self.device.type == 'cuda':
            self._cuda_stream = torch.cuda.Stream()
        
        # ★ 预热模型
        if self.model is not None and self.device.type == 'cuda':
            self._warmup_model()
        
        self.stats = {
            'total_processed': 0,
            'direct_valid': 0,
            'dl_corrected': 0,
            'rule_corrected': 0,
            'failed': 0
        }
    
    def _find_optimal_batch_size(self) -> int:
        """
        ★ 自动找到最优batch_size以最大化GPU利用率
        """
        if not TORCH_AVAILABLE or self.model is None or self.device.type != 'cuda':
            return self.training_config.inference_batch_size
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # 目标使用80%显存
        target_memory = int(total_memory * 0.8)
        
        # 从小batch开始测试
        test_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        optimal_size = 64
        
        # 创建测试输入
        test_seq_len = 128  # 假设平均序列长度
        
        for batch_size in test_sizes:
            try:
                torch.cuda.empty_cache()
                
                # 测试前向传播
                dummy_input = torch.randint(
                    0, 100, (batch_size, test_seq_len), 
                    dtype=torch.long, device=self.device
                )
                
                with torch.no_grad():
                    if self.use_fp16:
                        with torch.cuda.amp.autocast():
                            _ = self.model.generate_greedy_batch(dummy_input, max_length=test_seq_len)
                    else:
                        _ = self.model.generate_greedy_batch(dummy_input, max_length=test_seq_len)
                
                torch.cuda.synchronize()
                
                current_memory = torch.cuda.max_memory_allocated()
                
                if current_memory < target_memory:
                    optimal_size = batch_size
                else:
                    break
                    
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    break
                raise
        
        torch.cuda.empty_cache()
        logger.info(f"自动检测最优batch_size: {optimal_size}")
        return optimal_size
    
    def _warmup_model(self):
        """
        ★ 预热模型，确保CUDA kernel已编译
        """
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        logger.info("预热模型中...")
        try:
            dummy_input = torch.randint(
                0, 100, (4, 64), 
                dtype=torch.long, device=self.device
            )
            
            with torch.no_grad():
                for _ in range(3):  # 多次预热
                    if self.use_fp16:
                        with torch.cuda.amp.autocast():
                            _ = self.model.generate_greedy_batch(dummy_input, max_length=64)
                    else:
                        _ = self.model.generate_greedy_batch(dummy_input, max_length=64)
            
            torch.cuda.synchronize()
            logger.info("模型预热完成")
        except Exception as e:
            logger.debug(f"模型预热失败: {e}")
    
    def _validate_smiles(self, smiles: str) -> bool:
        """验证SMILES是否有效"""
        if not RDKIT_AVAILABLE or not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _canonicalize(self, smiles: str) -> Optional[str]:
        """标准化SMILES"""
        if not RDKIT_AVAILABLE or not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return None
    
    def _preprocess(self, smiles: str) -> Optional[str]:
        """预处理SMILES字符串"""
        if smiles is None:
            return None
        
        s = str(smiles).strip()
        
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        
        if not s or s.lower() in ['nan', 'none', 'null', 'na', 'n/a']:
            return None
        
        return s
    
    def _dl_correct(self, smiles: str, beam_size: int = 5) -> List[str]:
        """使用深度学习模型生成纠错候选（单个）"""
        if not TORCH_AVAILABLE or self.model is None:
            return []
        
        try:
            self.model.eval()
            
            src_ids = self.tokenizer.encode(smiles)
            src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
            
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    results = self.model.generate_beam(
                        src_tensor, beam_size=beam_size, max_length=256,
                        length_penalty=self.training_config.length_penalty, return_all=True
                    )
            else:
                results = self.model.generate_beam(
                    src_tensor, beam_size=beam_size, max_length=256,
                    length_penalty=self.training_config.length_penalty, return_all=True
                )
            
            candidates = []
            for seq, score in results[0]:
                decoded = self.tokenizer.decode(seq[0].tolist())
                if decoded:
                    candidates.append(decoded)
            
            return candidates
        
        except Exception as e:
            logger.debug(f"DL纠错失败: {e}")
            return []
    
    def _dl_correct_batch(self, smiles_list: List[str], beam_size: int = 5, 
                          use_greedy_first: bool = True) -> List[List[str]]:
        """
        ★ 批量深度学习纠错 - 深度优化版
        
        特性：
        1. 真正的批量GPU推理
        2. 异步数据传输
        3. 自动batch_size优化
        4. 最小化CPU-GPU同步
        5. ★ 支持任务取消检测
        """
        if not TORCH_AVAILABLE or self.model is None or not smiles_list:
            return [[] for _ in smiles_list]
        
        # ★ 检查取消标志
        try:
            from .task_manager import is_cancelled
            if is_cancelled():
                logger.info("DL纠错任务已被取消")
                return [[] for _ in smiles_list]
        except ImportError:
            pass
        
        try:
            self.model.eval()
            
            # 批量编码（CPU上进行）
            src_ids_list, lengths = self.tokenizer.batch_encode(smiles_list)
            
            # ★ 使用pin_memory加速传输
            src_np = np.array(src_ids_list, dtype=np.int64)
            if self.training_config.pin_memory and self.device.type == 'cuda':
                src_tensor = torch.from_numpy(src_np).pin_memory().to(self.device, non_blocking=True)
            else:
                src_tensor = torch.tensor(src_ids_list, dtype=torch.long, device=self.device)
            
            all_candidates = []
            
            # ★ 使用CUDA stream进行异步处理
            stream = self._cuda_stream if self._cuda_stream is not None else torch.cuda.current_stream() if self.device.type == 'cuda' else None
            
            with torch.no_grad():
                if stream is not None:
                    with torch.cuda.stream(stream):
                        all_candidates = self._run_inference(
                            src_tensor, smiles_list, beam_size, use_greedy_first
                        )
                    stream.synchronize()
                else:
                    all_candidates = self._run_inference(
                        src_tensor, smiles_list, beam_size, use_greedy_first
                    )
            
            return all_candidates
        
        except Exception as e:
            logger.warning(f"批量DL纠错失败: {e}")
            import traceback
            traceback.print_exc()
            return [[] for _ in smiles_list]
    
    def _run_inference(self, src_tensor: torch.Tensor, smiles_list: List[str],
                       beam_size: int, use_greedy_first: bool) -> List[List[str]]:
        """
        ★ 核心推理逻辑 - 支持FP16和批量处理
        """
        all_candidates = []
        
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                if use_greedy_first:
                    # ★ 贪婪解码 - 最快
                    greedy_results = self.model.generate_greedy_batch(src_tensor, max_length=256)
                    for i in range(len(smiles_list)):
                        decoded = self.tokenizer.decode(greedy_results[i].tolist())
                        all_candidates.append([decoded] if decoded else [])
                else:
                    # Beam Search - 更精确但较慢
                    beam_results = self.model.generate_beam_batch_optimized(
                        src_tensor, beam_size=beam_size, max_length=256,
                        length_penalty=self.training_config.length_penalty, return_all=True
                    )
                    for i, results in enumerate(beam_results):
                        candidates = []
                        for seq, score in results:
                            decoded = self.tokenizer.decode(seq[0].tolist())
                            if decoded:
                                candidates.append(decoded)
                        all_candidates.append(candidates)
        else:
            if use_greedy_first:
                greedy_results = self.model.generate_greedy_batch(src_tensor, max_length=256)
                for i in range(len(smiles_list)):
                    decoded = self.tokenizer.decode(greedy_results[i].tolist())
                    all_candidates.append([decoded] if decoded else [])
            else:
                beam_results = self.model.generate_beam_batch_optimized(
                    src_tensor, beam_size=beam_size, max_length=256,
                    length_penalty=self.training_config.length_penalty, return_all=True
                )
                for i, results in enumerate(beam_results):
                    candidates = []
                    for seq, score in results:
                        decoded = self.tokenizer.decode(seq[0].tolist())
                        if decoded:
                            candidates.append(decoded)
                    all_candidates.append(candidates)
        
        return all_candidates
    
    def _rule_correct(self, smiles: str) -> Optional[str]:
        """使用规则方法修复"""
        if not SMILES_UTILS_AVAILABLE:
            return None
        
        methods = [
            ('smart', lambda s: smart_repair_smiles(s, keep_largest_frag=True) if smart_repair_smiles else None),
            ('aggressive', lambda s: aggressive_repair_smiles(s) if aggressive_repair_smiles else None),
            ('ultra', lambda s: ultra_repair_smiles(s)[0] if ultra_repair_smiles else None),
        ]
        
        for name, method in methods:
            try:
                result = method(smiles)
                if result and self._validate_smiles(result):
                    return result
            except Exception as e:
                logger.debug(f"{name} 修复失败: {e}")
        
        return None
    
    def correct(self, smiles: str, 
               use_dl: bool = True,
               use_rules: bool = True,
               beam_size: int = 5,
               return_details: bool = False) -> Union[str, 'CorrectionResult']:
        """修复单个SMILES"""
        self.stats['total_processed'] += 1
        
        result = CorrectionResult(
            original=smiles, corrected=None, status='unknown',
            method=None, candidates=[], is_valid=False
        )
        
        cleaned = self._preprocess(smiles)
        if cleaned is None:
            result.status = 'invalid_input'
            self.stats['failed'] += 1
            return result if return_details else None
        
        if self._validate_smiles(cleaned):
            canonical = self._canonicalize(cleaned)
            result.corrected = canonical or cleaned
            result.status = 'valid'
            result.method = 'direct'
            result.is_valid = True
            self.stats['direct_valid'] += 1
            return result if return_details else result.corrected
        
        if use_dl and self.model is not None:
            candidates = self._dl_correct(cleaned, beam_size=beam_size)
            result.candidates = candidates
            
            for candidate in candidates:
                if self._validate_smiles(candidate):
                    canonical = self._canonicalize(candidate)
                    result.corrected = canonical or candidate
                    result.status = 'corrected'
                    result.method = 'transformer'
                    result.is_valid = True
                    self.stats['dl_corrected'] += 1
                    return result if return_details else result.corrected
        
        if use_rules:
            rule_result = self._rule_correct(cleaned)
            if rule_result:
                result.corrected = rule_result
                result.status = 'corrected'
                result.method = 'rule'
                result.is_valid = True
                self.stats['rule_corrected'] += 1
                return result if return_details else result.corrected
        
        result.status = 'failed'
        self.stats['failed'] += 1
        return result if return_details else None
    
    def correct_batch(self, smiles_list: List[str], 
                     use_dl: bool = True,
                     use_rules: bool = True,
                     beam_size: int = 5,
                     show_progress: bool = True,
                     return_details: bool = False,
                     batch_size: int = None,
                     progress_callback: callable = None) -> List[Union[str, 'CorrectionResult']]:
        """
        ★ 批量修复SMILES - 深度优化版
        
        特性：
        1. 真正的批量GPU推理
        2. 自动batch_size优化
        3. 异步数据传输
        4. 最小化CPU-GPU同步
        5. ★ 详细进度显示
        
        Args:
            progress_callback: 可选的回调函数，签名为 callback(current, total, stage, info_dict)
        """
        import time
        
        if not smiles_list:
            return []
        
        total_samples = len(smiles_list)
        start_time = time.time()
        
        # ★ 使用自动计算的最优batch_size
        if batch_size is None:
            batch_size = self._optimal_batch_size or self.training_config.inference_batch_size
        
        results = [None] * len(smiles_list)
        
        # ★ 进度信息
        progress_info = {
            'total': total_samples,
            'processed': 0,
            'valid_direct': 0,
            'needs_dl': 0,
            'dl_success': 0,
            'rule_success': 0,
            'failed': 0,
            'speed': 0.0,
            'eta': 0.0,
            'stage': '预处理',
            'batch_size': batch_size
        }
        
        def update_progress(stage, processed=None):
            if processed is not None:
                progress_info['processed'] = processed
            progress_info['stage'] = stage
            elapsed = time.time() - start_time
            if elapsed > 0 and progress_info['processed'] > 0:
                progress_info['speed'] = progress_info['processed'] / elapsed
                remaining = total_samples - progress_info['processed']
                progress_info['eta'] = remaining / progress_info['speed'] if progress_info['speed'] > 0 else 0
            
            if progress_callback:
                progress_callback(progress_info['processed'], total_samples, stage, progress_info)
        
        # ========== 阶段1：预处理和快速验证 ==========
        if show_progress:
            print(f"\n{'='*60}")
            print(f"🚀 SMILES批量纠错 - 共 {total_samples:,} 条")
            print(f"{'='*60}")
            print(f"📋 配置: batch_size={batch_size}, FP16={self.use_fp16}")
            print(f"\n[阶段 1/3] 预处理和验证...")
        
        preprocessed = []
        needs_correction_indices = []
        
        preprocess_iter = enumerate(smiles_list)
        if show_progress:
            preprocess_iter = tqdm(
                list(preprocess_iter), 
                desc="⏳ 预处理", 
                unit="条",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        for i, smiles in preprocess_iter:
            self.stats['total_processed'] += 1
            
            cleaned = self._preprocess(smiles)
            if cleaned is None:
                results[i] = CorrectionResult(
                    original=smiles, corrected=None, status='invalid_input',
                    method=None, candidates=[], is_valid=False
                ) if return_details else None
                self.stats['failed'] += 1
                progress_info['failed'] += 1
            elif self._validate_smiles(cleaned):
                canonical = self._canonicalize(cleaned)
                results[i] = CorrectionResult(
                    original=smiles, corrected=canonical or cleaned, status='valid',
                    method='direct', candidates=[], is_valid=True
                ) if return_details else (canonical or cleaned)
                self.stats['direct_valid'] += 1
                progress_info['valid_direct'] += 1
            else:
                preprocessed.append(cleaned)
                needs_correction_indices.append(i)
        
        progress_info['needs_dl'] = len(preprocessed)
        update_progress('预处理完成', progress_info['valid_direct'] + progress_info['failed'])
        
        if show_progress:
            print(f"   ✅ 直接有效: {progress_info['valid_direct']:,} 条")
            print(f"   🔧 需要纠错: {progress_info['needs_dl']:,} 条")
        
        # ========== 阶段2：批量深度学习纠错 ==========
        if use_dl and self.model is not None and preprocessed:
            if show_progress:
                print(f"\n[阶段 2/3] 深度学习纠错 (GPU)...")
            
            all_dl_candidates = []
            total_batches = (len(preprocessed) + batch_size - 1) // batch_size
            dl_processed = 0
            
            batch_iter = range(0, len(preprocessed), batch_size)
            if show_progress:
                batch_iter = tqdm(
                    list(batch_iter),
                    desc="🤖 DL纠错",
                    unit="批",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} 批 [{elapsed}<{remaining}] {postfix}',
                    postfix={'速度': '计算中...'}
                )
            
            batch_start_time = time.time()
            
            for batch_idx, start_idx in enumerate(batch_iter):
                end_idx = min(start_idx + batch_size, len(preprocessed))
                batch_smiles = preprocessed[start_idx:end_idx]
                current_batch_size = len(batch_smiles)
                
                batch_candidates = self._dl_correct_batch(
                    batch_smiles, beam_size=beam_size,
                    use_greedy_first=self.training_config.use_greedy_first
                )
                all_dl_candidates.extend(batch_candidates)
                
                dl_processed += current_batch_size
                
                # 更新进度条后缀
                if show_progress and hasattr(batch_iter, 'set_postfix'):
                    elapsed_batch = time.time() - batch_start_time
                    speed = dl_processed / elapsed_batch if elapsed_batch > 0 else 0
                    batch_iter.set_postfix({
                        '速度': f'{speed:.0f}条/秒',
                        '已处理': f'{dl_processed:,}/{len(preprocessed):,}'
                    })
                
                update_progress('DL纠错', progress_info['valid_direct'] + progress_info['failed'] + dl_processed)
            
            still_needs_rule = []
            still_needs_rule_indices = []
            
            # 验证DL结果
            if show_progress:
                print(f"   📊 验证DL纠错结果...")
            
            for i, (idx, candidates) in enumerate(zip(needs_correction_indices, all_dl_candidates)):
                original = smiles_list[idx]
                cleaned = preprocessed[i]
                
                found_valid = False
                for candidate in candidates:
                    if self._validate_smiles(candidate):
                        canonical = self._canonicalize(candidate)
                        results[idx] = CorrectionResult(
                            original=original, corrected=canonical or candidate, 
                            status='corrected', method='transformer', 
                            candidates=candidates, is_valid=True
                        ) if return_details else (canonical or candidate)
                        self.stats['dl_corrected'] += 1
                        progress_info['dl_success'] += 1
                        found_valid = True
                        break
                
                if not found_valid:
                    still_needs_rule.append(cleaned)
                    still_needs_rule_indices.append(idx)
            
            if show_progress:
                print(f"   ✅ DL纠错成功: {progress_info['dl_success']:,} 条")
                print(f"   ⏭️  需要规则修复: {len(still_needs_rule):,} 条")
        else:
            still_needs_rule = preprocessed
            still_needs_rule_indices = needs_correction_indices
        
        # ========== 阶段3：规则修复 ==========
        if use_rules and still_needs_rule:
            if show_progress:
                print(f"\n[阶段 3/3] 规则修复...")
            
            rule_iter = list(zip(still_needs_rule_indices, still_needs_rule))
            if show_progress:
                rule_iter = tqdm(
                    rule_iter, 
                    desc="📐 规则修复",
                    unit="条",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )
            
            for idx, cleaned in rule_iter:
                original = smiles_list[idx]
                rule_result = self._rule_correct(cleaned)
                
                if rule_result:
                    results[idx] = CorrectionResult(
                        original=original, corrected=rule_result, status='corrected',
                        method='rule', candidates=[], is_valid=True
                    ) if return_details else rule_result
                    self.stats['rule_corrected'] += 1
                    progress_info['rule_success'] += 1
                else:
                    results[idx] = CorrectionResult(
                        original=original, corrected=None, status='failed',
                        method=None, candidates=[], is_valid=False
                    ) if return_details else None
                    self.stats['failed'] += 1
                    progress_info['failed'] += 1
            
            if show_progress:
                print(f"   ✅ 规则修复成功: {progress_info['rule_success']:,} 条")
        elif still_needs_rule:
            for idx in still_needs_rule_indices:
                original = smiles_list[idx]
                results[idx] = CorrectionResult(
                    original=original, corrected=None, status='failed',
                    method=None, candidates=[], is_valid=False
                ) if return_details else None
                self.stats['failed'] += 1
                progress_info['failed'] += 1
        
        # ========== 最终汇总报告 ==========
        total_time = time.time() - start_time
        
        if show_progress:
            success_count = progress_info['valid_direct'] + progress_info['dl_success'] + progress_info['rule_success']
            success_rate = success_count / total_samples * 100 if total_samples > 0 else 0
            avg_speed = total_samples / total_time if total_time > 0 else 0
            
            print(f"\n{'='*60}")
            print(f"✅ 处理完成!")
            print(f"{'='*60}")
            print(f"📊 统计汇总:")
            print(f"   总计处理: {total_samples:,} 条")
            print(f"   总计耗时: {total_time:.2f} 秒")
            print(f"   平均速度: {avg_speed:.1f} 条/秒")
            print(f"\n📈 结果分布:")
            print(f"   ├─ 直接有效: {progress_info['valid_direct']:,} 条 ({progress_info['valid_direct']/total_samples*100:.1f}%)")
            print(f"   ├─ DL纠错:   {progress_info['dl_success']:,} 条 ({progress_info['dl_success']/total_samples*100:.1f}%)")
            print(f"   ├─ 规则修复: {progress_info['rule_success']:,} 条 ({progress_info['rule_success']/total_samples*100:.1f}%)")
            print(f"   └─ 失败:     {progress_info['failed']:,} 条 ({progress_info['failed']/total_samples*100:.1f}%)")
            print(f"\n🎯 成功率: {success_rate:.2f}%")
            print(f"{'='*60}\n")
        
        # 调用最终回调
        update_progress('完成', total_samples)
        
        return results
    
    def correct_dataframe(self, df: pd.DataFrame, 
                         smiles_column: str,
                         output_column: str = 'corrected_smiles',
                         status_column: Optional[str] = 'correction_status',
                         method_column: Optional[str] = 'correction_method',
                         use_dl: bool = True,
                         use_rules: bool = True,
                         beam_size: int = 5,
                         inplace: bool = False) -> pd.DataFrame:
        """修复DataFrame中的SMILES列"""
        if not inplace:
            df = df.copy()
        
        results = self.correct_batch(
            df[smiles_column].tolist(),
            use_dl=use_dl, use_rules=use_rules, beam_size=beam_size,
            show_progress=True, return_details=True
        )
        
        df[output_column] = [r.corrected for r in results]
        
        if status_column:
            df[status_column] = [r.status for r in results]
        
        if method_column:
            df[method_column] = [r.method for r in results]
        
        return df
    
    def train_from_valid_smiles(self, valid_smiles: List[str],
                                val_ratio: float = 0.1,
                                augment_factor: int = 5,
                                save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """从有效SMILES列表训练模型"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法训练模型")
        
        logger.info(f"🖥️ 训练设备: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"📊 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        logger.info(f"准备训练数据，原始样本: {len(valid_smiles)}")
        
        valid_smiles = [s for s in valid_smiles if s and self._validate_smiles(s)]
        logger.info(f"有效SMILES: {len(valid_smiles)}")
        
        if len(valid_smiles) < 10:
            raise ValueError(f"有效SMILES数量太少（{len(valid_smiles)}），至少需要10个")
        
        noise_injector = SMILESNoiseInjector(self.tokenizer, self.training_config)
        
        all_pairs = noise_injector.augment_dataset(valid_smiles, augment_factor=augment_factor)
        logger.info(f"生成训练对: {len(all_pairs)}")
        
        logger.info("正在构建词汇表...")
        for src_smiles, tgt_smiles in all_pairs:
            self.tokenizer.encode(src_smiles)
            self.tokenizer.encode(tgt_smiles)
        
        actual_vocab_size = self.tokenizer.vocab_size
        max_vocab_size = self.tokenizer.config.max_vocab_size
        logger.info(f"词汇表大小: {actual_vocab_size}, 配置最大值: {max_vocab_size}")
        
        if actual_vocab_size > max_vocab_size:
            logger.warning(f"词汇表({actual_vocab_size})超过配置最大值({max_vocab_size})，自动扩展")
            self.tokenizer.config.max_vocab_size = actual_vocab_size + 100
        
        np.random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * (1 - val_ratio))
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]
        
        model_vocab_size = actual_vocab_size + 100
        
        if self.model is not None:
            old_vocab_size = self.model.encoder.embedding.num_embeddings
            if old_vocab_size < actual_vocab_size:
                logger.warning(f"旧模型vocab_size({old_vocab_size})小于当前需要({actual_vocab_size})，重新创建模型")
                self.model = None
        
        if self.model is None:
            old_max = self.tokenizer.config.max_vocab_size
            self.tokenizer.config.max_vocab_size = max(model_vocab_size, old_max)
            self.model = SMILESCorrectorModel(self.transformer_config, self.tokenizer)
            self.tokenizer.config.max_vocab_size = old_max
            self.model.to(self.device)
            logger.info(f"创建模型，embedding大小: {self.model.encoder.embedding.num_embeddings}")
        
        self.training_config.device = str(self.device)
        trainer = SMILESCorrectorTrainer(self.model, self.tokenizer, self.training_config)
        history = trainer.train(train_pairs, val_pairs, save_path=save_path)
        
        return history
    
    def save_model(self, path: str):
        """保存模型和分词器"""
        if self.model is None:
            raise ValueError("没有模型可保存")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        if TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': self.model.config.__dict__,
                'transformer_config': self.transformer_config.__dict__,
                'training_config': self.training_config.__dict__
            }, path)
        
        tokenizer_path = path.replace('.pt', '_tokenizer.json')
        self.tokenizer.save(tokenizer_path)
        
        logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path: str):
        """加载模型和分词器，并自动优化配置"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法加载模型")
        
        tokenizer_path = path.replace('.pt', '_tokenizer.json')
        if os.path.exists(tokenizer_path):
            self.tokenizer = SMILESTokenizer.load(tokenizer_path)
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'transformer_config' in checkpoint:
            self.transformer_config = TransformerConfig(**checkpoint['transformer_config'])
        if 'training_config' in checkpoint:
            # 保留新的优化配置
            old_config = self.training_config
            self.training_config = TrainingConfig(**checkpoint['training_config'])
            # 恢复优化相关配置
            self.training_config.inference_batch_size = old_config.inference_batch_size
            self.training_config.use_fp16 = old_config.use_fp16
            self.training_config.auto_batch_size = old_config.auto_batch_size
        
        self.model = SMILESCorrectorModel(self.transformer_config, self.tokenizer)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # ★ FP16支持
        if self.training_config.use_fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
            self.use_fp16 = True
        
        # ★ 尝试torch.compile()
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                self._compiled = True
                logger.info("已启用torch.compile()加速")
            except Exception as e:
                logger.debug(f"torch.compile()不可用: {e}")
        
        # ★ 自动计算最优batch_size
        if self.training_config.auto_batch_size and self.device.type == 'cuda':
            self._optimal_batch_size = self._find_optimal_batch_size()
        
        # ★ 预热模型
        if self.device.type == 'cuda':
            self._warmup_model()
        
        logger.info(f"模型已从 {path} 加载")
        logger.info(f"推理配置: batch_size={self._optimal_batch_size or self.training_config.inference_batch_size}, "
                   f"FP16={self.use_fp16}, compiled={self._compiled}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = (stats['direct_valid'] + stats['dl_corrected'] + 
                                    stats['rule_corrected']) / stats['total_processed']
            stats['dl_contribution'] = stats['dl_corrected'] / stats['total_processed']
            stats['rule_contribution'] = stats['rule_corrected'] / stats['total_processed']
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_processed': 0,
            'direct_valid': 0,
            'dl_corrected': 0,
            'rule_corrected': 0,
            'failed': 0
        }
    
    def set_batch_size(self, batch_size: int):
        """
        ★ 手动设置推理批量大小
        
        如果自动检测的batch_size不理想，可以手动设置
        """
        self._optimal_batch_size = batch_size
        logger.info(f"推理batch_size已设置为: {batch_size}")
    
    def benchmark(self, n_samples: int = 1000, batch_sizes: List[int] = None,
                  seq_length: int = 100) -> Dict[str, Any]:
        """
        ★ 性能基准测试
        
        测试不同batch_size下的GPU利用率和吞吐量
        帮助找到最优配置
        
        Args:
            n_samples: 测试样本数量
            batch_sizes: 要测试的batch_size列表
            seq_length: 测试序列长度
        
        Returns:
            包含各配置性能数据的字典
        """
        if not TORCH_AVAILABLE or self.model is None:
            logger.error("模型未加载，无法进行基准测试")
            return {}
        
        if self.device.type != 'cuda':
            logger.warning("非CUDA设备，基准测试结果可能不准确")
        
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        
        results = {
            'device': str(self.device),
            'fp16': self.use_fp16,
            'compiled': self._compiled,
            'benchmarks': []
        }
        
        if self.device.type == 'cuda':
            results['gpu_name'] = torch.cuda.get_device_name(0)
            results['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("=" * 70)
        print("🚀 GPU性能基准测试")
        print("=" * 70)
        print(f"设备: {results.get('gpu_name', str(self.device))}")
        print(f"FP16: {self.use_fp16}, torch.compile: {self._compiled}")
        print(f"测试样本数: {n_samples}, 序列长度: {seq_length}")
        print("-" * 70)
        print(f"{'Batch Size':>12} | {'显存使用':>12} | {'GPU利用率':>12} | {'吞吐量':>15} | {'延迟':>12}")
        print("-" * 70)
        
        for batch_size in batch_sizes:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 生成测试数据
                dummy_input = torch.randint(
                    0, 100, (batch_size, seq_length),
                    dtype=torch.long, device=self.device
                )
                
                # 预热
                with torch.no_grad():
                    if self.use_fp16:
                        with torch.cuda.amp.autocast():
                            _ = self.model.generate_greedy_batch(dummy_input, max_length=seq_length)
                    else:
                        _ = self.model.generate_greedy_batch(dummy_input, max_length=seq_length)
                torch.cuda.synchronize()
                
                # 计时测试
                import time
                n_iterations = max(1, n_samples // batch_size)
                
                torch.cuda.reset_peak_memory_stats()
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    for _ in range(n_iterations):
                        if self.use_fp16:
                            with torch.cuda.amp.autocast():
                                _ = self.model.generate_greedy_batch(dummy_input, max_length=seq_length)
                        else:
                            _ = self.model.generate_greedy_batch(dummy_input, max_length=seq_length)
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                
                # 统计
                memory_used = torch.cuda.max_memory_allocated() / 1e9
                throughput = (batch_size * n_iterations) / elapsed
                latency = elapsed / n_iterations * 1000  # ms
                
                # 获取GPU利用率（近似）
                memory_ratio = memory_used / (results.get('total_memory_gb', 8))
                
                benchmark_result = {
                    'batch_size': batch_size,
                    'memory_gb': memory_used,
                    'memory_ratio': memory_ratio,
                    'throughput': throughput,
                    'latency_ms': latency
                }
                results['benchmarks'].append(benchmark_result)
                
                print(f"{batch_size:>12} | {memory_used:>10.2f}GB | {memory_ratio*100:>10.1f}% | "
                      f"{throughput:>12.1f}/s | {latency:>10.2f}ms")
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"{batch_size:>12} | {'OOM':>12} | {'-':>12} | {'-':>15} | {'-':>12}")
                    torch.cuda.empty_cache()
                    break
                raise
        
        print("-" * 70)
        
        # 推荐最优配置
        if results['benchmarks']:
            best = max(results['benchmarks'], key=lambda x: x['throughput'])
            print(f"\n✅ 推荐batch_size: {best['batch_size']}")
            print(f"   吞吐量: {best['throughput']:.1f} samples/sec")
            print(f"   显存使用: {best['memory_gb']:.2f} GB ({best['memory_ratio']*100:.1f}%)")
            
            results['recommended_batch_size'] = best['batch_size']
        
        print("=" * 70)
        
        torch.cuda.empty_cache()
        return results
    
    def diagnose_gpu(self):
        """
        ★ GPU诊断 - 检查GPU配置和潜在问题
        """
        print("=" * 60)
        print("🔍 GPU诊断报告")
        print("=" * 60)
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch未安装")
            return
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，请检查驱动和安装")
            return
        
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  总显存: {props.total_memory / 1e9:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器数: {props.multi_processor_count}")
        
        # 当前显存状态
        print(f"\n当前显存状态:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # 模型状态
        print(f"\n模型状态:")
        print(f"  模型已加载: {self.model is not None}")
        print(f"  FP16启用: {self.use_fp16}")
        print(f"  torch.compile: {self._compiled}")
        print(f"  当前batch_size: {self._optimal_batch_size or self.training_config.inference_batch_size}")
        
        # 建议
        print(f"\n💡 优化建议:")
        if self._optimal_batch_size and self._optimal_batch_size < 256:
            print(f"  - 当前batch_size({self._optimal_batch_size})较小，尝试增大")
        if not self.use_fp16:
            print(f"  - 建议启用FP16以提升速度和减少显存")
        if not self._compiled:
            print(f"  - 如果PyTorch>=2.0，建议启用torch.compile")
        
        print("=" * 60)


@dataclass
class CorrectionResult:
    """纠错结果"""
    original: str
    corrected: Optional[str]
    status: str
    method: Optional[str]
    candidates: List[str]
    is_valid: bool


# =============================================================================
# 便捷函数
# =============================================================================

def create_smiles_correction_pipeline(model_path: Optional[str] = None,
                                      device: str = 'auto') -> SMILESCorrectionPipeline:
    """创建SMILES纠错流水线"""
    pipeline = SMILESCorrectionPipeline(device=device)
    
    if model_path and os.path.exists(model_path):
        pipeline.load_model(model_path)
    
    return pipeline


def correct_smiles(smiles: str, 
                   model_path: Optional[str] = None,
                   use_dl: bool = True,
                   use_rules: bool = True) -> Optional[str]:
    """便捷函数：修复单个SMILES"""
    pipeline = create_smiles_correction_pipeline(model_path)
    return pipeline.correct(smiles, use_dl=use_dl, use_rules=use_rules)


def correct_smiles_batch(smiles_list: List[str],
                        model_path: Optional[str] = None,
                        use_dl: bool = True,
                        use_rules: bool = True) -> List[Optional[str]]:
    """便捷函数：批量修复SMILES"""
    pipeline = create_smiles_correction_pipeline(model_path)
    return pipeline.correct_batch(smiles_list, use_dl=use_dl, use_rules=use_rules)


# =============================================================================
# 测试和演示
# =============================================================================

def _demo():
    """演示SMILES纠错流水线"""
    print("=" * 60)
    print("SMILES Transformer 纠错器演示 (优化版)")
    print("=" * 60)
    
    test_smiles = [
        'CCO',
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CC(=O)OC1=CC=CC=C1C(=O)O)',
        'CC(=O)OC1=CC=CC=C1C(=O)O[Na+]',
        'CC1=CC=C(C=C1)C(C)C',
        'CCN(CC)CC',
        'c1ccccc',
        '[nH]1cccc1',
        'invalid_smiles',
    ]
    
    pipeline = SMILESCorrectionPipeline()
    
    print("\n测试SMILES纠错（规则方法）:")
    print("-" * 60)
    
    for smiles in test_smiles:
        result = pipeline.correct(smiles, use_dl=False, use_rules=True, return_details=True)
        print(f"\n输入: {smiles}")
        print(f"输出: {result.corrected}")
        print(f"状态: {result.status}")
        print(f"方法: {result.method}")
    
    print("\n" + "=" * 60)
    print("统计信息:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    _demo()
