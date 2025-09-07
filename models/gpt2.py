from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPT2Config:
  vocab_size: int
  d_model: int = 512
  max_seq_len: int = 1024

class SingleHeadSelfAttention(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super(SingleHeadSelfAttention, self).__init__()

    self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
    self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
    self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)

  def _causal_mask(self, seq_len: int):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

  def forward(self, x: torch.Tensor):
    B, T, C = x.shape

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)

    mask = self._causal_mask(T).to(x.device)
    attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

    attn_probs = F.softmax(attn_weights, dim=-1)

    out = torch.matmul(attn_probs, v)

    return out

class FeedForward(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super(FeedForward, self).__init__()

    self.net = nn.Sequential(
      nn.Linear(cfg.d_model, 2 * cfg.d_model),
      nn.GELU(),
      nn.Linear(2 * cfg.d_model, cfg.d_model),
    )

  def forward(self, x: torch.Tensor):
    return self.net(x)

class TransformerLayer(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super(TransformerLayer, self).__init__()

    self.norm1 = nn.LayerNorm(cfg.d_model)
    self.linear1 = nn.Linear(cfg.d_model, cfg.d_model)

    self.self_attn = SingleHeadSelfAttention(cfg)
    self.linear2 = nn.Linear(cfg.d_model, cfg.d_model)

    self.norm2 = nn.LayerNorm(cfg.d_model)
    self.ff = FeedForward(cfg)

  def forward(self, x: torch.Tensor):
    x = self.norm1(x)
    x = self.linear1(x)

    x = x + self.linear2(self.self_attn(x))

    x = x + self.ff(self.norm2(x))

    return x

class GPT2(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super(GPT2, self).__init__()
    self.cfg = cfg

    self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
    self.wpe = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    self.layer = TransformerLayer(cfg)

    self.norm = nn.LayerNorm(cfg.d_model)

    self.ln_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

  def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    B, T = x.shape

    pos = torch.arange(0, T).view(1, -1).to(x.device)
    x = self.wte(x) + self.wpe(pos) # (B, T, d_model)

    x = self.layer(x)

    x = self.norm(x)
    x = self.ln_head(x)

    return x

  def generate(self, idx: torch.Tensor, max_new_tokens: int = 5):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.cfg.max_seq_len:]
      T = idx_cond.shape[1]

      y = self(idx_cond)
      y = y[:, -1, :]
      probs = F.softmax(y, dim=-1)

      next_token = torch.multinomial(probs, num_samples=1)

      idx = torch.cat((idx, next_token), dim=1)
    return idx