import torch
import tiktoken
from torch.utils.data import Dataset

class FileTextDataset(Dataset):
  def __init__(self, file_path: str, tokenizer: tiktoken.Encoding, block_size: int = 128):
    self.tokenizer = tokenizer
    self.block_size = block_size

    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    self.token_ids = self.tokenizer.encode(text)

  def __len__(self):
    return len(self.token_ids) - self.block_size

  def __getitem__(self, idx):
    x = torch.tensor(self.token_ids[idx:idx+self.block_size])
    y = torch.tensor(self.token_ids[idx+1:idx+self.block_size+1])
    return x, y

