import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, model_name: str):
  os.makedirs('checkpoints', exist_ok=True)

  for fname in os.listdir('checkpoints'):
    if fname.startswith(model_name):
      fpath = os.path.join('checkpoints', fname)
      if os.path.isfile(fpath):
        os.remove(fpath)

  ckpt_path = os.path.join('checkpoints', f'{model_name}_checkpoint_epoch_{epoch}.pth')
  torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
  }, ckpt_path)

  print(f'Checkpoint saved to {ckpt_path}')

def load_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], ckpt_path: str):
  ckpt = torch.load(ckpt_path)

  model.load_state_dict(ckpt['model_state'])
  if optimizer is not None:
    optimizer.load_state_dict(ckpt['optimizer_state'])

  print(f'Loaded checkpoint from {ckpt_path} (epoch {ckpt['epoch']})')
  return ckpt['epoch']