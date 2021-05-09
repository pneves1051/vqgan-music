import math
import numpy as np
import torch
import torch.nn as nn
from models.vq_vae.modules import VQVAEEncoder, VQVAEDecoder, VectorQuantizer
from models.vq_vae.attention import SelfAttn, AttnBlock


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


class AttnVQVAE(nn.Module):
  def __init__(self, embed_dim, n_embed, input_channels, output_channels, comp, depth):
    super(AttnVQVAE, self).__init__()
    self.embed_dim = embed_dim
    self.n_embed = n_embed

    kernel_size = 8
    stride=4
    padding = (kernel_size-stride)//2

    self.input_channels = input_channels
    self.output_channels = output_channels

    num_layers = int(math.log(comp, stride))
    
    self.pre_encoder = [nn.Conv1d(input_channels, embed_dim, kernel_size,stride=stride, padding=padding), nn.ReLU()]
    for _ in range(num_layers-2):
      self.pre_encoder.extend([nn.Conv1d(embed_dim, embed_dim, kernel_size,stride=stride, padding=padding), nn.ReLU()])
    self.pre_encoder.extend([nn.Conv1d(embed_dim, embed_dim, kernel_size,stride=stride, padding=padding)])
    self.pre_encoder = nn.Sequential(*self.pre_encoder)
    
    self.encoder = AttnBlock(input_channels=input_channels,
                            input_axis=1,
                            num_freq_bands = 6,
                            max_freq = 44100,
                            depth = depth,
                            num_latents = 512,
                            latent_dim = embed_dim,
                            cross_heads = 1,
                            latent_heads = 8,
                            cross_dim_head = 64,
                            latent_dim_head = 64,
                            num_classes = 1000,
                            attn_dropout = 0.,
                            ff_dropout = 0.,
                            weight_tie_layers = False,
                            fourier_encode_data = True,
                            self_per_cross_attn = 2)

    self.vector_quantizer = VectorQuantizer(embed_dim, n_embed)
    
    self.decoder = AttnBlock(input_channels=embed_dim,
                            input_axis=1,
                            num_freq_bands = 6,
                            max_freq = 44100,
                            depth = depth,
                            num_latents = 512,
                            latent_dim = embed_dim,
                            cross_heads = 1,
                            latent_heads = 8,
                            cross_dim_head = 64,
                            latent_dim_head = 64,
                            num_classes = 1000,
                            attn_dropout = 0.,
                            ff_dropout = 0.,
                            weight_tie_layers = False,
                            fourier_encode_data = True,
                            self_per_cross_attn = 2)

    self.post_decoder = []
    for _ in range(num_layers-1):
      self.post_decoder.extend([nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size, stride=stride, padding=padding), nn.ReLU()])
    self.post_decoder = nn.Sequential(*self.post_decoder)
    
    self.last_conv = nn.ConvTranspose1d(embed_dim, output_channels, kernel_size, stride=stride, padding=padding)

    self.tanh = nn.Tanh()
    
  # returns the vectors zq, ze and the indices
  def encode(self, inputs):
    #inputs_one_hot = F.one_hot(inputs, self.in_ch).permute(0, 2, 1).float()

    # input comes in format (batch, channels, len), so we have to transpose
    encoding = self.pre_encoder(inputs).transpose(-1, -2)
    encoding = self.encoder(encoding, inputs.transpose(-1, -2))
     
    quant, codes, indices = self.vector_quantizer(encoding)
        
    return encoding, quant, codes, indices

  def decode(self, quant):
    reconstructed = self.decoder(quant, quant).transpose(-1, -2)
    reconstructed = self.post_decoder(reconstructed)
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
