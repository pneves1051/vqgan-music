import torch
from torch import nn
import torch.nn.functional as F

from models.vq_vae.attention import Block, audio_upsample, audio_downsample, SelfAttn
from utils.utils import trunc_normal_

# Adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
class VectorQuantizer(nn.Module):
  def __init__(self, embed_dim, n_embed, decay=0.99, eps=1e-5, threshold=1.0):
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
    self.decay = decay
    self.eps = eps
    # d: dimension of each embedding latent vector
    # (k embedding vectors)
    # codebook: contains the k d-dimensional vectors from the quantized latent space
    embed = torch.randn(embed_dim, n_embed)
    self.register_buffer("embed", embed)
    self.register_buffer("cluster_size", torch.zeros(n_embed))
    self.register_buffer("embed_avg", embed.clone())
    
    self.threshold = threshold
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

    # EMA and restart
    if self.training:
      # find number of times each code occurred
      # (n_embed,)
      encodings_sum = encodings.sum(0)
      # each aij is the sum of all positions i of each element(vector)
      #  in the input to which codebook element j was the closest
      # each column j is the sum of all vectors in the input that where
      # closest to codebook element j 
      # (embed_dim, batch*len)@(batch*len, n_embed) -> (embed_dim, n_embed)
      # 
      embed_sum = flat_inputs.transpose(0, 1) @ encodings

      # EMA vectors below threshold to random ones in encoding 
      rand_inp = flat_inputs[torch.randperm(flat_inputs.shape[0])][:self.n_embed]

      #print(encodings_sum.shape, usage.shape)
      # EMA statistics of number of times each code occurred accounting for new vectors
      self.cluster_size.data.mul_(self.decay).add_(
        encodings_sum, alpha=1-self.decay)      
      # self.cluster_size.data.mul_(self.decay).add_(
        # encodings_sum, alpha=1-self.decay) 
        
      # calculate EMA and apply random restart
      self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)   
      # self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)     
      
      # check if usage falls below threshold
      usage = (self.cluster_size >= self.threshold).float().unsqueeze(1)

      n = self.cluster_size.sum()
      # cluster size scaled
      cluster_size = (
        (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
      )
     
      # So all vectors stay in same scale
      embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
      self.embed.data.copy_(usage.T*embed_normalized + ((1-usage)*rand_inp).T)

    # before the detach
    codes = torch.cat([inputs, quantized], axis=-1)

    # straight through estimator
    quantized = inputs + (quantized - inputs).detach()
    avg_probs = torch.mean(encodings, 0)
    
    return quantized, codes, encoding_indices
  
  def quantize(self, encoding_indices):
    """Returns embedding vector for a batch of indices"""
    return F.embedding(encoding_indices, self.embed.transpose(1,0))


class WSConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class WSConvTranspose1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv_transpose1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def Normalization(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class ResLayer(nn.Module):
  def __init__(self, chs, dilation, leaky=False, normalization=Normalization, conv=WSConv1d):
    super(ResLayer, self).__init__()
    padding = dilation
    self.conv = nn.Sequential(
                  normalization(chs),
                  #nn.BatchNorm1d(chs),
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  conv(chs, chs, 3, dilation=dilation, padding=padding),
                  #nn.Conv1d(chs, chs, 3, dilation=dilation, padding=padding),
                  normalization(chs),
                  #nn.BatchNorm1d(chs),
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  conv(chs, chs, 3, dilation=dilation, padding=padding)
                  #nn.Conv1d(chs, chs, 1, padding=0)
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
  def __init__(self, chs, dilations, depth, leaky=False, normalization = Normalization, conv=WSConv1d):
    super(ResBlock, self).__init__()
    self.res_block = nn.Sequential(*[ResLayer(chs, dilations[i], leaky, normalization=normalization, conv=conv) for i in range(depth)])

  def forward(self, x):
    out = self.res_block(x)
    return out

class VQVAEEncoder(nn.Module):
  def __init__(self, in_ch, out_ch, num_chs, strides, depth, leaky=False, normalization=Normalization, conv=WSConv1d):
    super(VQVAEEncoder,self).__init__()
    dilations = [3**i for i in range(depth)]

    # possível camada para condicionamento
    # self.cond = nn.AdaptiveMaxpool()
    
    kernel_size = 8
    stride=4
    padding = (kernel_size-stride)//2

    self.first_conv = conv(in_ch, num_chs[0], 3, padding=1)#nn.Conv1d(in_ch, num_chs[0], 3, padding=1)

    res_list = []
    for i in range(1, len(num_chs)):
      s = strides[i-1]
      res_list.append(nn.ModuleList([
                                    normalization(num_chs[i-1]),
                                    #nn.BatchNorm1d(num_chs[i-1]),
                                    nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                    conv(num_chs[i-1], num_chs[i], s*2, stride=s, padding=s//2),
                                    #nn.Conv1d(num_chs[i-1], num_chs[i], s*2, stride=s, padding=s//2),
                                    ResBlock(num_chs[i], dilations, depth, leaky, normalization=normalization, conv=conv)]))

    self.res_layers =  nn.ModuleList(res_list)

    # post processing layers
    self.last_res = nn.Sequential(normalization(num_chs[-1]),
                                  #nn.BatchNorm1d(num_chs[-1]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[-1], dilations, depth, leaky, normalization=normalization, conv=conv),
                                  SelfAttn(num_chs[-1], normalization=normalization, conv=conv),
                                  normalization(num_chs[-1]),
                                  #nn.BatchNorm1d(num_chs[-1]),                                  
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[-1], dilations, depth, leaky, normalization=normalization, conv=conv))
                                    
    
    self.last_conv = nn.Sequential(normalization(num_chs[-1]),
                              #nn.BatchNorm1d(num_chs[-1]),
                              nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                              nn.Conv1d(num_chs[-1], out_ch, 3, padding=1))
    
    # Attention blocks
    #self.attn_indices = attn_indices
    #self.attn_modules = nn.ModuleList([SelfAttn(num_chs[f], 'relu') for f in self.attn_indices])

  
  def forward(self, x):
    x = self.first_conv(x)
    j = 0
    for i, module in enumerate(self.res_layers):
      x = module[0](x)
      x = module[1](x)
      x = module[2](x)
      x = module[3](x)
      #if i + 1 in self.attn_indices:
      #  x = self.attn_modules[j](x)
      #  j += 1

    out = self.last_res(x)   
    out = self.last_conv(x)
    return out

class VQVAEDecoder(nn.Module):
  def __init__(self, in_ch, out_ch, num_chs, strides, depth, leaky=False, normalization=Normalization, conv=WSConv1d, conv_t=WSConvTranspose1d):
    super(VQVAEDecoder,self).__init__()
    dilations = [3**i for i in range(depth)]
    dilations = dilations[::-1]

    kernel_size = 8
    stride = 4
    output_padding = 0
    padding = (kernel_size + output_padding-stride)//2

    self.first_conv = conv(in_ch, num_chs[0], 3, padding=1)#nn.Conv1d(in_ch, num_chs[0], 3, padding=1)

    self.first_res = nn.Sequential(normalization(num_chs[0]),
                                  #nn.BatchNorm1d(num_chs[0]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[0], dilations, depth, leaky, normalization=normalization, conv=conv),
                                  SelfAttn(num_chs[0], normalization=normalization, conv=conv),
                                  normalization(num_chs[0]),
                                  #nn.BatchNorm1d(num_chs[0]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                                  ResBlock(num_chs[0], dilations, depth, leaky, normalization=normalization, conv=conv))

    res_list = []
    for i in range(1, len(num_chs)):
      s = strides[i-1]
      res_list.append(nn.ModuleList([ResBlock(num_chs[i-1], dilations, depth, leaky, normalization=normalization, conv=conv),
                      normalization(num_chs[i-1]),
                      #nn.BatchNorm1d(num_chs[i-1]),
                      nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                      conv_t(num_chs[i-1], num_chs[i], s*2, stride=s, padding=s//2)]))
                      #nn.ConvTranspose1d(num_chs[i-1], num_chs[i], s*2, stride=s, padding=s//2)]))

    self.res_layers =  nn.ModuleList(res_list)

    # post processing layers
    self.last_act = nn.Sequential(normalization(num_chs[-1]),
                                  #nn.BatchNorm1d(num_chs[-1]),
                                  nn.LeakyReLU(0.2) if leaky else nn.ReLU())

    self.last_conv = nn.Conv1d(num_chs[-1], out_ch, 3, padding=1)

    #self.attn_indices = attn_indices
    #self.attn_modules = nn.ModuleList([SelfAttn(num_chs[f], 'relu') for f in self.attn_indices])  
  
  
  def forward(self, x):
    x = self.first_conv(x)
    x = self.first_res(x)    
    j = 0
    for i, module in enumerate(self.res_layers):
      x = module[0](x)
      x = module[1](x)
      x = module[2](x)
      x = module[3](x)
      #if i + 1  in self.attn_indices:
      #  x = self.attn_modules[j](x)
      #  j += 1
       
    out = self.last_conv(self.last_act(x))
    return out


########ATTN########

class AttnEncoder(nn.Module):
  def __init__(self, embed_dim, sample_length, patch_size=256, in_channels=1, out_channels=512,  num_classes=10, depths=[2, 3, 3, 5],
                  num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  drop_path_rate=0., hybtrid_backbone=None, norm_layer=nn.LayerNorm):
    super(AttnEncoder, self).__init__()
    self.ch = embed_dim
    self.embed_dim = embed_dim
    self.sample_length = sample_length
    self.levels = len(depths)

    #self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
    
    #self.l1 = nn.Linear(latent_dim, (self.bottom_width) * self.embed_dim)
    self.pos_embed = [nn.Parameter(torch.zeros(1, self.sample_length//(4**i), embed_dim//(4**(len(depths)-1-i)))) for i in range(len(depths))]
    
    is_mask = True
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[0])] # stochastic depth decay rule

    self.blocks = nn.ModuleList([
                    Block(dim=embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for i in range(depths[0])])

    # List of sublists of blocks. Each sublist works in a specific scale.
    self.downsample_blocks = nn.ModuleList([
                    nn.ModuleList(
                      #[Block(dim=embed_dim//(4**i), num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      #drop_rate=drop_rate, attn_drop=drop_rate, drop_path=0, norm_layer=norm_layer)] +\
                      [Block(dim=embed_dim//(4**(len(depths)-1-i)), num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=drop_rate, drop_path=0, norm_layer=norm_layer, is_mask=0) for _ in range(depth)]) for i, depth in enumerate(depths)])

    for i in range(len(self.pos_embed)):
      trunc_normal_(self.pos_embed[i], std=.02)
                    
    self.deconv = nn.Sequential(nn.Conv1d(self.embed_dim, 1, 1, 1, 0))

  def forward(self, x, epoch=0):
    #x = self.patch_embed(x)
    B = x.size()
    #for index, block in enumerate(self.blocks):
    #  x = block(x, epoch)
    for index, block in enumerate(self.downsample_blocks):
      if index != 0:
        x, L = audio_downsample(x)
      x = x + self.pos_embed[index].to(x.get_device())
      for b in block:
        x = b(x, epoch)
          
    out = x
    return out


class AttnDecoder(nn.Module):
  def __init__(self, embed_dim, sample_length, patch_size=256, in_channels=1, num_classes=10, depths=[5, 3, 3, 2],
                  num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  drop_path_rate=0., hybtrid_backbone=None, norm_layer=nn.LayerNorm):
    super(AttnDecoder, self).__init__()
    self.ch = embed_dim
    self.embed_dim = embed_dim
    self.sample_length= sample_length
    self.levels = len(depths)

    #self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
    
    #self.l1 = nn.Linear(latent_dim, (self.sample_length) * self.embed_dim)
    self.pos_embed = [nn.Parameter(torch.zeros(1, int(self.sample_length//(4**(len(depths)-1-i))), embed_dim//(4**i))) for i in range(len(depths))]

    is_mask = True
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[0])] # stochastic depth decay rule

    self.blocks = nn.ModuleList([
                    Block(dim=embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for i in range(depths[0])])

    # List of sublists of blocks. Each sublist works in a specific scale.
    self.upsample_blocks = nn.ModuleList([
                    nn.ModuleList(
                      #[Block(dim=embed_dim//(4**i), num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      #drop_rate=drop_rate, attn_drop=drop_rate, drop_path=0, norm_layer=norm_layer)] +\
                      [Block(dim=embed_dim//(4**i), num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=drop_rate, drop_path=0., norm_layer=norm_layer, is_mask=0) for _ in range(depth)]) for i, depth in enumerate(depths)])

    for i in range(len(self.pos_embed)):
      trunc_normal_(self.pos_embed[i], std=.02)
                    
    self.deconv = nn.Sequential(nn.Conv1d(self.embed_dim//(4), 1, 1, 1, 0))

  def forward(self, x, epoch=0):
    #x = self.patch_embed(x)
    B = x.size()
    #for index, block in enumerate(self.blocks):
    #  x = block(x, epoch)
    for index, block in enumerate(self.upsample_blocks):
      if index != 0:
        x, L = audio_upsample(x)
      x = x + self.pos_embed[index].to(x.get_device())
      for b in block:
        x = b(x, epoch)

    #out = self.deconv(x.permute(0, 2, 1).view(-1, self.embed_dim//(4**self.levels), L)) 
    out = x
    return out

