import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
  def __init__(self, embed_dim, n_embed, decay=0.99, eps=1e-5):
    """
    Args:
      embed_dim: dimensionality of the tensors in the quantized space.
        Inputs to the modules must be in this format as well.
      n_embed: number of vectors in the quantized space.
    """
    super(VectorQuantizer, self).__init__()
    # k: size of the discrete latent space
    self.embed_dim = embed_dim
    self.n_embed = n_embed
    self.dacay = decay
    self.eps = eps
    # d: dimension of each embedding latent vector
    # (k embedding vectors)
    # codebook: contains the k d-dimensional vectors from the quantized latent space
    embed = torch.randn(dim, n_embed)
    self.register_buffer("embed", embed)
    self.register_buffer("cluster_size", torch.zeros(n_embed))
    self.register_buffer("embed_avg", embed.clone())
    
    #emb = torch.empty(embed_dim, n_embed)
    #emb.data.uniform_(-1/n_embed, 1/n_embed)
    #torch.nn.init.xavier_uniform_(emb)
    #self.embedding= nn.Parameter(emb)
    #self.register_parameter('embed', self.embedding)  

    #embed = nn.Embedding(n_embed, embed_dim)
  
  def forward(self, inputs):
    """Connects the module to some inputs.
    Args:
      inputs: Tensor, second dimension must be equal to embed_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.
   
   Returns:
        quantize: Tensor containing the quantized version of the input.
        encodings: Tensor containing the discrete encodings, ie which element
        of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
    """
    # input (batch, len, dim) -> (batch*len, embed_dim)    
    flat_inputs = inputs.reshape(-1, self.embed_dim)
    # distance between the input and the embedding elements (batch*len, embed_dim)
    distances = (
        torch.sum(flat_inputs**2, 1, keepdim=True)
        - 2*torch.matmul(flat_inputs, self.embed)
         + torch.sum(self.embed**2, 0, keepdim=True)
    )
    
    # index with smaller distance beetween the input and the embedding elements
    # (batch*len,)
    encoding_indices = torch.argmax(-distances, dim=1)
    # transform the index into one_hot to multiply by the embedding
    # (batch*len, n_embed)
    encodings = F.one_hot(encoding_indices, self.n_embed).type_as(distances)
    # multiply the index to find the quantization
    # (batch, len,)
    encoding_indices = encoding_indices.view(*inputs.shape[:-1])
    #print(set(encoding_indices.flatten().tolist()))
    # (batch, len, embed_dim)
    quantized = self.quantize(encoding_indices)

    if self.training:
      # find number of times each code occurred
      # (n_embed,)
      encodings_sum = encodings.sum(0)
      # dn
      # (embed_dim, batch*len)@(batch*len, n_embed) -> (embed_dim, n_embed)
      embed_sum = flat_inputs.transpose(0, 1) @ encodings

      self.cluster_size.data.mul_(self.decay).add_(
        encodings_sum, alpha=1-self.decay
      )
      # calculate EMA
      self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
      n = self.cluster_size.sum()
      cluster_size = (
        (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
      )
      embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
      self.embed.data.copy_(embed_normalized)

    # before the detach
    codes = torch.cat([inputs, quantized], axis=-1)

    # straight through estimator
    quantized = inputs + (quantized - inputs).detach()
    avg_probs = torch.mean(encodings, 0)
    
    return quantized, codes, encoding_indices
  
  def quantize(self, encoding_indices):
    """Returns embedding vector for a batch of indices"""
    return F.embedding(encoding_indices, self.embed.transpose(1,0))

class ResLayer(nn.Module):
  def __init__(self, chs, dilation, leaky=False):
    super(ResLayer, self).__init__()
    padding = dilation
    self.conv = nn.Sequential(
                  nn.BatchNorm1d(chs),
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  nn.Conv1d(chs, chs, 3, dilation=dilation, padding=padding),
                  nn.BatchNorm1d(chs),
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  nn.Conv1d(chs, chs, 1, padding=0),
                  )
           
  def forward(self, x):
    res_x = self.conv(x)
    out = x + res_x
    return out

class WNResLayer(nn.Module):
  def __init__(self, chs, dilation, leaky=False):
    super(ResLayer, self).__init__()
    padding = dilation
    self.conv = nn.Sequential(
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  nn.utils.weight_norm(nn.Conv1d(chs, chs, 3, dilation=dilation, padding=padding)),
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  nn.utils.weight_norm(nn.Conv1d(chs, chs, 1, padding=0)),
                  )
           
  def forward(self, x):
    res_x = self.conv(x)
    out = x + res_x
    return out

class ResBlock(nn.Module):
  def __init__(self, chs, dilations, depth, leaky=False):
    super(ResBlock, self).__init__()
    self.res_block = nn.Sequential(*[ResLayer(chs, dilations[i], leaky) for i in range(depth)])

  def forward(self, x):
    out = self.res_block(x)
    return out

class SelfAttn(nn.Module):
  def __init__(self, ch):
    super(SelfAttn, self).__init__()
    self.ch = ch
    
    # Key
    self.theta = nn.Conv1d(self.ch, self.ch//8, 1, bias = False)
    self.phi = nn.Conv1d(self.ch, self.ch//8, 1, bias = False)
    self.g = nn.Conv1d(self.ch, self.ch//2, 1, bias=False)
    self.o = nn.Conv1d(self.ch//2, self.ch, 1, bias=False)
    
    # Gain parameter
    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

  def forward(self, x):

    # query
    theta = self.theta(x)
    # key
    phi = F.max_pool1d(self.phi(x), [2])
    # value
    g = F.max_pool1d(self.g(x), [2])

    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)))

    return self.gamma * o + x
  
class VQVAEEncoder(nn.Module):
  def __init__(self, in_ch, out_ch, num_chs, strides, depth, attn_indices, leaky=False):
    super(VQVAEEncoder,self).__init__()
    dilations = [3**i for i in range(depth)]

    # possível camada para condicionamento
    # self.cond = nn.AdaptiveMaxpool()
    
    kernel_size = 8
    stride=4
    padding = (kernel_size-stride)//2

    self.first_conv = nn.Conv1d(in_ch, num_chs[0], 3, padding=1)

    res_list = []
    for i in range(1, len(num_chs)):
      s = strides[i-1]
      res_list.append(nn.ModuleList([nn.BatchNorm1d(num_chs[i-1]),
                                    nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                    nn.Conv1d(num_chs[i-1], num_chs[i], s*2, stride=s, padding=s//2),
                                    ResBlock(num_chs[i], dilations, depth, leaky)]))

    self.res_layers =  nn.ModuleList(res_list)

    # post processing layers
    self.last_res = nn.Sequential(nn.BatchNorm1d(num_chs[-1]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[-1], dilations, depth, leaky),
                                  SelfAttn(num_chs[-1]),
                                  nn.BatchNorm1d(num_chs[-1]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[-1], dilations, depth, leaky))
                                    
    
    self.last_conv = nn.Sequential(nn.BatchNorm1d(num_chs[-1]),
                              nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                              nn.Conv1d(num_chs[-1], out_ch, 3, padding=1))
    
    # Attention blocks
    self.attn_indices = attn_indices
    self.attn_modules = nn.ModuleList([SelfAttn(num_chs[f], 'relu') for f in self.attn_indices])

  
  def forward(self, x):
    x = self.first_conv(x)
    j = 0
    for i, module in enumerate(self.res_layers):
      x = module[0](x)
      x = module[1](x)
      x = module[2](x)
      x = module[3](x)
      if i + 1 in self.attn_indices:
        x = self.attn_modules[j](x)
        j += 1

    out = self.last_res(x)   
    out = self.last_conv(x)
    return out

class VQVAEDecoder(nn.Module):
  def __init__(self, in_ch, out_ch, num_chs, strides, depth, attn_indices, leaky=False):
    super(VQVAEDecoder,self).__init__()
    dilations = [3**i for i in range(depth)]
    dilations = dilations[::-1]

    kernel_size = 8
    stride = 4
    output_padding = 0
    padding = (kernel_size + output_padding-stride)//2

    self.first_conv = nn.Conv1d(in_ch, num_chs[0], 3, padding=1)

    self.first_res = nn.Sequential(nn.BatchNorm1d(num_chs[0]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[0], dilations, depth, leaky),
                                  SelfAttn(num_chs[0]),
                                  nn.BatchNorm1d(num_chs[0]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[0], dilations, depth, leaky))

    res_list = []
    for i in range(1, len(num_chs)):
      s = strides[i-1]
      res_list.append(nn.ModuleList([ResBlock(num_chs[i-1], dilations, depth, leaky),
                      nn.BatchNorm1d(num_chs[i-1]),
                      nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                      nn.ConvTranspose1d(num_chs[i-1], num_chs[i], s*2, stride=s, padding=s//2)]))

    self.res_layers =  nn.ModuleList(res_list)

    # post processing layers
    self.last_act = nn.Sequential(nn.BatchNorm1d(num_chs[-1]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU())

    self.last_conv = nn.Conv1d(num_chs[-1], out_ch, 3, padding=1)

    self.attn_indices = attn_indices
    self.attn_modules = nn.ModuleList([SelfAttn(num_chs[f], 'relu') for f in self.attn_indices])  
  
  
  def forward(self, x):
    x = self.first_conv(x)
    x = self.first_res(x)    
    j = 0
    for i, module in enumerate(self.res_layers):
      x = module[0](x)
      x = module[1](x)
      x = module[2](x)
      x = module[3](x)
      if i + 1  in self.attn_indices:
        x = self.attn_modules[j](x)
        j += 1
       
    out = self.last_conv(self.last_act(x))
    return out


  
  
