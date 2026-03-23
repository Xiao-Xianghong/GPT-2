from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math

@dataclass
class GPTConfig:
    block_size: int = 1024 # maximum context length
    vocab_size: int = 50257 # size of vocabulary library 50000 BPE tokens + 256 bytes tokens + 1 <|endoftext|> token 
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # dimensionality of token embeddings
    batch_size: int = 12
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))
        # ones[block_size, block_size] -> tril -> [1, 1, block_size, block_size]

    def forward(self, x):
        B, T, C = x.size() #batch_size, token_length, embd_dim
        qkv = self.c_attn(x) # [B, T, C] -> [B, T, 3 * C]
        q, k, v = qkv.split(self.n_embd, dim=2) # [B, T, 3 * C] -> 3 * [B, T, C]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, C // n_head]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, C // n_head]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, C // n_head]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [B, n_head, T, T]
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # softmax over the LAST DIMENSION, not the last two dims
        # which means normalize the attention weights for each token, not for all tokens
        y = att @ v # [B, n_head, T, C // n_head]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        y = self.c_proj(y) # [B, T, C] interaction between different heads
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd), 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #[T]
        pos_embd = self.transformer.wpe(pos) #[T, n_embd]
        tok_embd = self.transformer.wte(idx) #[B, T, n_embd]
        x = tok_embd + pos_embd # [B, T, n_embd]
        for block in self.transformer.h:
            x = block(x) # [B, T, n_embd]
        x = self.transformer.ln_f(x) # [B, T, n_embd]
        logits = self.lm_head(x) # [B, T, vocab_size]
        if targets is None:
            loss = None
        else: 
            B, T, vocab_size = logits.size()
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    #def generate