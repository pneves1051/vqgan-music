# Patch based multiscale discriminator
import torch
import torchaudio
torchaudio.set_audio_backend("sox_io")
import torch.nn as nn
import torch.nn.functional as F


class ModuleDiscriminator(nn.Module):
  def __init__(self, input_channels, num_filters, window_size, cont):
    super(ModuleDiscriminator, self).__init__()
    shuffle_n = 0
 
    kernel_size = 24
    stride=4
    padding = (kernel_size-stride)//2
    
    self.pre = nn.Sequential(nn.Conv1d(input_channels+1, num_filters[0], 9, 
                                      stride=1, padding=4),
                              nn.LeakyReLU(0.2))
    
    module_list = []
    for i in range(1, len(num_filters)):
      module_list.append(nn.Sequential(nn.Conv1d(num_filters[i-1], num_filters[i], kernel_size, padding=padding, stride=stride),
                        nn.LeakyReLU(0.2)))
    
    self.discriminator = nn.ModuleList(module_list)

    self.post = nn.Sequential(nn.Linear(num_filters[-1]*(window_size//cont), 1))
    
 
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
    output = self.post(h.flatten(1))
    results.append(output)    

    return results[:-1], results[-1]

class MultiDiscriminator(nn.Module):
  def __init__(self, input_channels, num_classes, num_filters, num_d, window_size, cont):
    super(MultiDiscriminator, self).__init__()
 
    self.num_filters = num_filters
    self.num_d = num_d
 
    self.cond = nn.Sequential(nn.Embedding(num_classes, 64),
                              nn.Linear(64, window_size),
                              Reshape((1, window_size)))

    
    self.discriminators = nn.ModuleList(
        [ModuleDiscriminator(input_channels, num_filters, window_size, cont*(2**(i)))
         for i in range(num_d)]
    )
 
    self.pooling = nn.ModuleList((
        [nn.Identity()]+
        [nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False) for _ in range(1, num_d)]
    ))
       
 
  def forward(self, x, labels):
    labels = torch.argmax(labels, dim=1)

    h = x     
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
