import math
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

from performer_pytorch import FastAttention

#Class responsible for the positional encoding
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len = 5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe=torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(position*div_term)
    pe[:, 1::2] = torch.cos(position*div_term)
    pe = pe.unsqueeze(0).transpose(0,1)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads, nb_features=256, causal=True):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size//heads

    self.attn_fn = FastAttention(dim_heads=self.head_dim, nb_features=nb_features, causal=causal)

    assert (self.head_dim*heads == embed_size), "Embed_size needs to be divisible by heads"

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
  
  def forward(self, queries, keys, values, mask=None):
    N = queries.shape[0]
    query_len, key_len, value_len = queries.shape[1], keys.shape[1], values.shape[1]

    # split embedding into self.heads pieces
    queries = queries.reshape(N, query_len, self.heads, self.head_dim).transpose(1, 2)
    queries = self.queries(queries)
   
    keys = keys.reshape(N, key_len, self.heads, self.head_dim).transpose(1, 2)
    keys = self.keys(keys)
        
    values = values.reshape(N, value_len, self.heads, self.head_dim).transpose(1,2)
    values = self.values(values) 

    attn = self.attn_fn(queries, keys, values)

    attn = attn.transpose(1, 2).reshape(N, query_len, self.heads*self.head_dim)

    out = self.fc_out(attn)
    return out

class EncoderBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion, nb_features=256, causal=True):
    super(EncoderBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads, nb_features=nb_features, causal=causal)

    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.ff = nn.Sequential(
      nn.Linear(embed_size, forward_expansion*embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion*embed_size, embed_size)
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, queries, keys, values):
    attn = self.attention(queries, keys, values)
    x = self.dropout(self.norm1(attn + queries))
    forward = self.ff(x)
    out = self.dropout(self.norm2(forward + x))

    return out

class TransformerEncoder(nn.Module):
  def __init__(self, num_layers, embed_size, heads, dropout, forward_expansion, nb_features=256, causal=True):
    super(TransformerEncoder, self).__init__()
    
    self.blocks = nn.ModuleList([EncoderBlock(embed_size, heads, dropout, forward_expansion,
                                                 nb_features, causal) for _ in range(num_layers)])
     
  def forward(self, x):
    for block in self.blocks:
      x = block(x, x, x)

    return x
  


