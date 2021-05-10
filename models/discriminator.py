# Patch based multiscale discriminator
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq_vae.attention import AttnBlock



class ModuleDiscriminator(nn.Module):
  def __init__(self, in_ch, num_chs, stride, window_size, cont):
    super(ModuleDiscriminator, self).__init__()
    shuffle_n = 0
 
    self.pre = nn.Sequential(nn.Conv1d(in_ch, num_chs[0], 9, 
                                      stride=1, padding=4),
                              nn.LeakyReLU(0.2))
    
    module_list = []
    for i in range(1, len(num_chs)):
      module_list.append(nn.Sequential(nn.Conv1d(num_chs[i-1], num_chs[i],
                                                                      kernel_size=stride * 10 + 1,
                                                                      stride=stride,
                                                                      padding=stride * 5,
                                                                      groups=num_chs[i-1] // 4),                       
                        nn.LeakyReLU(0.2)))
    
    module_list.append(nn.Conv1d(num_chs[-1], 1, kernel_size=3, stride=1, padding=1))
    
    self.discriminator = nn.ModuleList(module_list)

    #self.post = nn.Sequential(nn.Linear(num_chs[-1]*(window_size//cont), 1))
     
  def forward(self, x):
    results = []
    h = x
    h = self.pre(h)
    #print(h.shape)
    results.append(h)

    for module in self.discriminator:
      h = module(h)
      #print(h.shape)
      results.append(h)
    #output = self.post(h.flatten(1))
    #results.append(output)    

    return results[:-1], results[-1]

class MultiDiscriminator(nn.Module):
  def __init__(self, in_ch, num_chs, stride, num_d, window_size, cont, n_classes=None):
    super(MultiDiscriminator, self).__init__()
 
    self.in_ch = in_ch+1 if n_classes is not None else in_ch
    self.num_chs = num_chs
    self.num_d = num_d
    self.n_classes = n_classes    
    
    if n_classes is not None:

      self.cond = nn.Sequential(nn.Embedding(n_classes, 64),
                                nn.Linear(64, window_size),
                                Reshape((1, window_size)))

    
    self.discriminators = nn.ModuleList(
        [ModuleDiscriminator(self.in_ch, num_chs, stride, window_size, cont*(2**(i)))
         for i in range(num_d)]
    )
 
    self.pooling = nn.ModuleList((
        [nn.Identity()]+
        [nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False) for _ in range(1, num_d)]
    ))
       
 
  def forward(self, x, labels=None):
    h = x
    if self.n_classes is not None and labels is not None:
      labels = torch.argmax(labels, dim=1)
      y = self.cond(labels)
      h = torch.cat([h, y], dim=1)
    
    #ann_cond = self.ann_cond(annotations)
    #lyrics_cond = self.lyrics_cond(lyrics)
    #x = torch.cat([x, lyrics_cond, ann_cond], 1)
    results = []
    for pool, disc in zip(self.pooling, self.discriminators):
      h = pool(h)
      results.append(disc(h))
      
    return results  # (feat, score), (feat, score), ...


class AttnDiscriminator(nn.Module):
  def __init__(self, embed_dim, input_channels, comp, depth):
    super().__init__()

    kernel_size = 8
    stride=4
    padding = (kernel_size-stride)//2

    self.input_channels = input_channels
    
    num_layers = int(math.log(comp, stride))

    self.pre_encoder = [nn.Conv1d(input_channels, embed_dim, kernel_size,stride=stride, padding=padding), nn.LeakyReLU(0.2)]
    for _ in range(num_layers-2):
      self.pre_encoder.extend([nn.Conv1d(embed_dim, embed_dim, kernel_size,stride=stride, padding=padding), nn.LeakyReLU(0.2)])
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
  def forward(self, inputs, conditions=None):
    results = []
    x = self.pre_encoder(inputs).transpose(-1, -2)
    out = self.encoder(x, inputs.transpose(-1, -2))

    results.append((x, out))

    return results