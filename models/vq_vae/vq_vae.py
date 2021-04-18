import numpy as np
import torch
import torch.nn as nn
from models.vq_vae.modules import VQVAEEncoder, VQVAEDecoder, VectorQuantizer

class VQVAE(nn.Module):
  def __init__(self, embed_dim, n_embed, in_ch, out_ch, num_chs, strides, depth, attn_indices):
    super(VQVAE, self).__init__()
    self.embed_dim = embed_dim
    self.n_embed = n_embed

    self.in_ch = in_ch
    self.out_ch=out_ch

    enc_attn_indices = []#attn_indices
    dec_attn_indices = []#[(len(num_chs)-1)-i for i in attn_indices]

    self.encoder = VQVAEEncoder(in_ch, embed_dim, num_chs, strides, depth, enc_attn_indices)
  
    self.vector_quantizer = VectorQuantizer(embed_dim, n_embed)
    
    self.decoder = VQVAEDecoder(embed_dim, out_ch, num_chs[::-1], strides[::-1], depth, dec_attn_indices)

    self.tanh = nn.Tanh()
    
  # returns os the vectors zq, ze and the indices
  def encode(self, inputs):
    #inputs_one_hot = F.one_hot(inputs, self.in_ch).permute(0, 2, 1).float()

    encoding = self.encoder(inputs)
     
    quant, codes, indices = self.vector_quantizer(encoding.permute(0, 2, 1))
    quant = quant.permute(0, 2, 1)
    
    return encoding, quant, codes, indices

  def decode(self, quant):
    reconstructed = self.decoder(quant)
    reconstructed = self.tanh(reconstructed)

    return reconstructed

  # a way to get the codebook
  def get_vqvae_codebook(self):
    codebook = self.vector_quantizer.quantize(torch.arange(self.n_embed, device=next(self.parameters()).device))
    codebook = codebook.reshape(self.n_embed, self.embed_dim)

    return codebook

  def get_last_layer(self):
    return self.decoder.last_conv.weight

  def forward(self, inputs):
    encoding, quant, codes, indices = self.encode(inputs)

    reconstructed = self.decode(quant)    

    return reconstructed, codes
