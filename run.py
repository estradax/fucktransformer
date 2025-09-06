from dataclasses import dataclass
from typing import Optional

import tiktoken
from tiktoken import Encoding

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(420)

@dataclass
class GPT2Config:
  vocab_size: int
  d_model: int = 512
  max_seq_len: int = 1024

class SingleHeadSelfAttention(nn.Module):
  def __init__(self, d_model):
    super(SingleHeadSelfAttention, self).__init__()

    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)

  def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    B, T, C = x.shape

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
    if mask is not None:
      attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

    attn_probs = F.softmax(attn_weights, dim=-1)

    out = torch.matmul(attn_probs, v)

    return out

class TransformerLayer(nn.Module):
  def __init__(self, d_model: int):
    super(TransformerLayer, self).__init__()

    self.norm1 = nn.LayerNorm(d_model)
    self.linear1 = nn.Linear(d_model, d_model)

    self.self_attn = SingleHeadSelfAttention(d_model)

    self.norm2 = nn.LayerNorm(d_model)
    self.ff = nn.Linear(d_model, d_model)

  def forward(self, x: torch.Tensor):
    x = self.norm1(x)
    x = self.linear1(x)

    x = x + self.self_attn(x)

    x = self.norm2(x)
    x = x + self.ff(x)

    return x

class GPT2(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super(GPT2, self).__init__()

    self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
    self.wpe = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    self.layer = TransformerLayer(cfg.d_model)

    self.norm = nn.LayerNorm(cfg.d_model)

    self.ln_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

  def forward(self, x: torch.Tensor):
    B, T = x.shape

    pos = torch.arange(0, T).view(1, -1)
    x = self.wte(x) + self.wpe(pos) # (B, T, d_model)

    x = self.layer(x)

    x = self.norm(x)
    x = self.ln_head(x)
    x = F.softmax(x, dim=-1)

    return x

def get_sample_data(enc: Encoding, txt: str):
  return torch.tensor(enc.encode(txt)).view(1, -1)

if __name__ == '__main__':
  enc = tiktoken.get_encoding('gpt2')
  dat = get_sample_data(enc, 'Hello, my name is')

  config = GPT2Config(vocab_size=enc.n_vocab)
  gpt2 = GPT2(config)

  logits = gpt2(dat)
  print(logits.shape)