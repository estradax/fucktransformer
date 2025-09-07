import torch
import tiktoken

from models.gpt2 import GPT2, GPT2Config
from utils.utils import load_checkpoint

torch.manual_seed(420)

if __name__ == '__main__':
  enc = tiktoken.get_encoding('gpt2')

  config = GPT2Config(vocab_size=enc.n_vocab)
  gpt2 = GPT2(config)
  load_checkpoint(gpt2, None, 'checkpoints/gpt2_checkpoint_epoch_5.pth')

  gpt2.eval()
  with torch.no_grad():
    dat = torch.tensor(enc.encode('Indonesia, dengan nama resmi Republik Indonesia,')).unsqueeze(0)
    generated = gpt2.generate(dat, max_new_tokens=100)

    print(enc.decode_batch(generated.tolist()))