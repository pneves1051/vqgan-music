import math
import numpy as np
import torch
import torch.nn as nn
from models.vq_vae.modules import VQVAEEncoder, VQVAEDecoder, VectorQuantizer, AttnEncoder, AttnDecoder, ResBlock


class VQVAE(nn.Module):
  def __init__(self, embed_dim, n_embed, in_ch, out_ch, num_chs, strides, depth, attn_indices, threshold = 0.0):
    super(VQVAE, self).__init__()
    self.embed_dim = embed_dim
    self.n_embed = n_embed

    self.in_ch = in_ch
    self.out_ch=out_ch

    enc_attn_indices = []#attn_indices
    dec_attn_indices = []#[(len(num_chs)-1)-i for i in attn_indices]

    self.encoder = VQVAEEncoder(in_ch, embed_dim, num_chs, strides, depth, enc_attn_indices)
  
    self.vector_quantizer = VectorQuantizer(embed_dim, n_embed, threshold=threshold)
    
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


class AttnVQVAE(nn.Module):
  def __init__(self, embed_dim, n_embed, input_channels, output_channels, sample_length, depths, pre_levels):
    super(AttnVQVAE, self).__init__()
    self.embed_dim = embed_dim
    self.n_embed = n_embed

    dilations = [3**i for i in range(3)]
    depth = 3

    kernel_size = 8
    stride=4
    padding = (kernel_size-stride)//2

    self.input_channels = input_channels
    self.output_channels = output_channels

    self.first_conv = nn.Conv1d(input_channels, embed_dim//(4**(len(depths)-1)), 3, padding=1)
    enc_res_list = []
    for i in range(pre_levels):
      enc_res_list.extend([nn.BatchNorm1d(embed_dim//(4**(len(depths)-1))),
                           nn.ReLU(),
                           nn.Conv1d(embed_dim//(4**(len(depths)-1)), embed_dim//(4**(len(depths)-1)), kernel_size, stride=stride, padding=padding),
                           ResBlock(embed_dim//(4**(len(depths)-1)), dilations, depth)])
    self.enc_res = nn.Sequential(*enc_res_list)
    
    self.encoder = AttnEncoder(embed_dim, sample_length=sample_length//(stride**pre_levels), in_channels=input_channels, depths=depths)
    
    self.vector_quantizer = VectorQuantizer(embed_dim, n_embed)
    
    self.decoder = AttnDecoder(embed_dim, sample_length=sample_length//(stride**pre_levels), in_channels=input_channels, depths=depths[::-1])

    dec_res_list = []
    for i in range(pre_levels):
      dec_res_list.extend([ResBlock(embed_dim//(4**(len(depths)-1)), dilations, depth),
                           nn.BatchNorm1d(embed_dim//(4**(len(depths)-1))),
                           nn.ReLU(),
                           nn.ConvTranspose1d(embed_dim//(4**(len(depths)-1)), embed_dim//(4**(len(depths)-1)), kernel_size, stride=stride, padding=padding)])
    self.dec_res = nn.Sequential(*dec_res_list)
    
    self.last_conv = nn.Conv1d(embed_dim//(4**(len(depths)-1)), output_channels, 3, padding=1)

    self.tanh = nn.Tanh()
    
  # returns the vectors zq, ze and the indices
  def encode(self, inputs):
    #inputs_one_hot = F.one_hot(inputs, self.in_ch).permute(0, 2, 1).float()
    x = self.first_conv(inputs)
    x = self.enc_res(x).transpose(-1, -2)

    # input comes in format (batch, channels, len), so we have to transpose
    encoding = self.encoder(x)
    
    quant, codes, indices = self.vector_quantizer(encoding)
    
    # print(inputs.shape, quant.shape)
        
    return encoding, quant, codes, indices

  def decode(self, quant):
    reconstructed = self.decoder(quant).transpose(-1, -2)
    reconstructed = self.dec_res(reconstructed)
    reconstructed = self.last_conv(reconstructed)
    reconstructed = self.tanh(reconstructed)

    return reconstructed

  # a way to get the codebook
  def get_vqvae_codebook(self):
    codebook = self.vector_quantizer.quantize(torch.arange(self.n_embed, device=next(self.parameters()).device))
    codebook = codebook.reshape(self.n_embed, self.embed_dim)

    return codebook

  def get_last_layer(self):
    return self.last_conv.weight

  def forward(self, inputs):
    encoding, quant, codes, indices = self.encode(inputs)

    reconstructed = self.decode(quant)    

    return reconstructed, codes
