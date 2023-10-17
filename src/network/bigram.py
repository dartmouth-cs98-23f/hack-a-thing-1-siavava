#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

# hyperparameters

batch_size = 32     # number of batches to process in parallel.
block_size = 8      # maximum context length.
epochs = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_size = 32
# ------------
with open("data/input.txt", "r") as f:
  text = f.read()

STOI = {ch: i for i, ch in enumerate(sorted(set(text)))}
ITOS = {i: ch for ch, i in STOI.items()}
vocab_size = len(STOI)

def encode(text: str):
  return [STOI[ch] for ch in text]

def decode(indices: list):
  return ''.join(ITOS[i] for i in indices)


data = torch.tensor(encode(text))

split = int(len(data) * 0.8)
train_data = data[:split]
val_data = data[split:]

torch.manual_seed(1337)       # Set the random seed for reproducibility
CONTEXT_LENGTH = 8            # Maximum context length.
BATCH_SIZE = 4                # Number of independent sequences to train on in parallel

def get_batch(split: str): 
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size: int):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_size)
    self.lm_head = nn.Linear(embedding_size, vocab_size)

  def forward(self, idx, targets=None):

    token_embeddings = self.embeddings(idx)
    logits = self.lm_head(token_embeddings)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    """
      idx: (B, T)
    """

    for _ in range(max_new_tokens):
      logits, loss = self(idx)

      logits = logits[:, -1, :]

      probs = F.softmax(logits, dim=-1)

      idx_next = torch.multinomial(probs, num_samples=1)

      idx = torch.cat([idx, idx_next], dim=-1)

    return idx
  
model = BigramLanguageModel(vocab_size)
model = model.to(device)

idx = torch.zeros( (1, 1), dtype=torch.long)
decode(model.generate(idx, max_new_tokens=100)[0].tolist())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

batch_size = 32

for epoch in range(epochs):

  if epoch % eval_interval == 0:
    losses = estimate_loss()
    print(f"Epoch {epoch:5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  # if steps % 100 == 0:
  #   print(f"Step: {steps:4} Loss: {loss.item():.4f}")

print(decode(model.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=400)[0].tolist()))

print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
