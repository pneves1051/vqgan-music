import math
import torch
import torch.nn as nn
from models.transformers.transformer_modules import PositionalEncoding

class Transformer(nn.Module):
  def __init__(self, vocab_size, d_model, n_head, n_hid, n_layer, max_len, dropout = 0.1):
    super(Transformer, self).__init__()
    self.model_type = 'Transformer'
    self.vocab_size = vocab_size    
    self.d_model = d_model
        
    self.src_mask = None
    self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
    self.embedding = nn.Embedding(vocab_size, d_model)
    #self.embedding.load_state_dict({'weight': codebook})
    #self.embedding.requires_grad=False

    encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, n_hid, dropout=dropout)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer)
    self.decoder = nn.Linear(d_model, vocab_size)
    
    self.init_weights()
    
  def generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

  def init_weights(self):
    initrange =0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)
    
  def forward(self, input, mask):    
    src = self.embedding(input)*math.sqrt(self.d_model)
    src = self.pos_encoder(src)
        
    src = self.transformer_encoder(src, mask)
    output = self.decoder(src)

    return output
