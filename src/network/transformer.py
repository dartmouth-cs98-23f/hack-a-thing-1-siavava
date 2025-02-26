#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from data import get_batch, encode, decode, vocab_size

import yaml
with open('config.yml', 'r') as f:
  config = yaml.safe_load(f)
  for k, v in config.items():
    globals()[k] = v

# # hyperparameters
# batch_size        = config.get("batch_size") # how many independent sequences will we process in parallel?
# block_size        = config.get("block_size") # what is the maximum context length for predictions?
# max_iters         = config.get("max_iters")
# eval_interval     = config.get("eval_interval")
# learning_rate     = config.get("learning_rate")
# eval_iters        = config.get("eval_iters")
# embeddings_size   = config.get("embeddings_size")
# head_count        = config.get("head_count")
# n_layer           = config.get("n_layer")
# dropout           = config.get("dropout")
# seed              = config.get("seed")
# # # ------------

torch.manual_seed(seed)

@torch.no_grad()
def estimate_loss(model: nn.Module):
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

class Head(nn.Module):
  """One self-attention head."""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(embeddings_size, head_size, bias=False)
    self.query = nn.Linear(embeddings_size, head_size, bias=False)
    self.value = nn.Linear(embeddings_size, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B,T,C = x.shape
    k = self.key(x)   # (B,T,hs)
    q = self.query(x) # (B,T,hs)
    # compute attention scores ("affinities")

    # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    weights= q @ k.transpose(-2,-1) * k.shape[-1]**-0.5

    # (B, T, T)
    weights= weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    # (B, T, T)
    weights= F.softmax(weights, dim=-1)
    weights= self.dropout(weights)
    # perform the weighted aggregation of the values

      # (B,T,hs)
    values = self.value(x)

      # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    out = weights @ values
    return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(head_size * num_heads, embeddings_size)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out

class FeedFoward(nn.Module):
  """
    Feed-forward network with ReLU activation and dropout.
    This is applied to each position separately and identically. 
  """

  def __init__(self, embeddings_size):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(embeddings_size, 4 * embeddings_size),
      nn.ReLU(),
      nn.Linear(4 * embeddings_size, embeddings_size),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.network(x)

class Block(nn.Module):
  """
    One transformer block.
    Consists of self-attention, layer normalization, feed-forward, and residual connections.

    Note: for simplicity, we omit the skip-connection from the paper.
  """

  def __init__(self, embeddings_size, head_count):
    # embeddings_size: embedding dimension, head_count: the number of heads we'd like
    super().__init__()
    head_size = embeddings_size // head_count
    self.sa = MultiHeadAttention(head_count, head_size)
    self.ffwd = FeedFoward(embeddings_size)
    self.ln1 = nn.LayerNorm(embeddings_size)
    self.ln2 = nn.LayerNorm(embeddings_size)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class GPTLanguageModel(nn.Module):
  """
    The full GPT language model, with a single embedding layer, n layers, and a linear output layer.

    Note: for simplicity, we omit the skip-connection from the paper.

    Note: we don't apply the logits mask here, that is done externally.
  """

  def __init__(self):
      super().__init__()
      # each token directly reads off the logits for the next token from a lookup table
      self.token_embedding_table = nn.Embedding(vocab_size, embeddings_size)
      self.position_embedding_table = nn.Embedding(block_size, embeddings_size)
      self.blocks = nn.Sequential(*[Block(embeddings_size, head_count=head_count) for _ in range(n_layer)])
      self.ln_f = nn.LayerNorm(embeddings_size) # final layer norm
      self.lm_head = nn.Linear(embeddings_size, vocab_size)

      # better init, not covered in the original GPT video, but important, will cover in followup video
      self.apply(self._init_weights)

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, index, targets=None):
    
    B, T = index.shape

    # index and targets are both (B,T) tensor of integers
    token_embeddings = self.token_embedding_table(index)                                    #? (B,T,C)
    positional_embeddings = self.position_embedding_table(torch.arange(T, device=device))   #? (T,C)
    x = token_embeddings + positional_embeddings                                            #? (B,T,C)
    x = self.blocks(x)                                                                      #? (B,T,C)
    x = self.ln_f(x)                                                                        #? (B,T,C)
    logits = self.lm_head(x)                                                                #? (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, index, max_new_tokens):
    """
      `index`: a (B, T) array of indices in the current context

      `max_new_tokens`: the maximum number of tokens to generate after `index`
    """
    for _ in range(max_new_tokens):

      # crop index to last block_size tokens
      index_cond = index[:, -block_size:]

      # get predictions
      logits, _ = self(index_cond)                          #? ignore loss

      # focus only on the last time step
      logits = logits[:, -1, :]                             #? reshape to (B, C)

      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1)                     #? (B, C)

      # sample from distribution
      index_next = torch.multinomial(probs, num_samples=1)  #? (B, 1)

      # append sampled index to the running sequence
      index = torch.cat((index, index_next), dim=1)         #? (B, T+1)
      
    return index

  def train_model(self, save_name: str, checkpoint=0):
    """
      Train the model.
    """

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

    for iter in range(checkpoint, max_iters):
      print(f"{iter}/{max_iters}", end='\r')

      # every once in a while evaluate the loss on train and val sets
      if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(self)
        print(f"step {iter:5}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        torch.save(self.state_dict(), f'models/{save_name}-{iter}.pth')

      # sample a batch of data
      xb, yb = get_batch('train')

      # evaluate the loss
      logits, loss = self(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

    # save model
    torch.save(self.state_dict(), f'models/{save_name}.pth')

def load_model(path=None):
  """
    load the latest model checkpoint.
  """

  if not path:
    path = "models/transfusion.pth"
  model = GPTLanguageModel()

  # load last checkpoint
  model.load_state_dict(torch.load(path, map_location=device))

  # cast model to device (CPU or GPU)
  model = model.to(device)

  return model



__all__ = [
    "estimate_loss"
  , "GPTLanguageModel"
  , "encode"
  , "decode"
  , "load_model"
]
