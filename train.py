import torch
import torch.optim as optim
import torch.nn as nn
import tiktoken

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from models.gpt2 import GPT2, GPT2Config
from utils.utils import save_checkpoint
from utils.dataset import FileTextDataset

torch.manual_seed(420)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=0.0):
  def lr_lambda(current_step):
      if current_step < warmup_steps:
          return float(current_step) / float(max(1, warmup_steps))

      progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
      return max(min_lr, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))))
  return LambdaLR(optimizer, lr_lambda)

if __name__ == '__main__':
  lr = 1e-3
  batch_size = 4
  epochs = 5
  device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
  print(f'Using device: {device}')

  enc = tiktoken.get_encoding('gpt2')
  cfg = GPT2Config(vocab_size=enc.n_vocab)

  dataset = FileTextDataset('sample_data/indonesia_wikipedia.txt', tokenizer=enc, block_size=cfg.max_seq_len)
  dataloader = DataLoader(dataset, batch_size=batch_size)

  model = GPT2(cfg)
  optimizer = optim.AdamW(model.parameters(), lr=lr)
  scheduler = ReduceLROnPlateau(optimizer)
  criterion = nn.CrossEntropyLoss()

  model.to(device)
  model.train()

  for epoch in range(epochs):
    total_loss = 0.0
    for step, (x, y) in enumerate(dataloader):
      x, y = x.to(device), y.to(device)
      logits = model(x)
      loss = criterion(logits.view(-1, logits.shape[-1]), y.view(-1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      if step % 10 == 0:
        print(f'Step {step}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')

    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)

    print(f'Epoch {epoch+1}, Average Loss: {avg_loss}, LR: {scheduler.get_last_lr()[0]}')
    save_checkpoint(model, optimizer, epoch+1, 'gpt2')
