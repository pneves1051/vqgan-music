import math
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from torch.nn.modules import pixelshuffle
from torch.nn.modules.pixelshuffle import PixelShuffle

from utils.utils import DropPath

def Normalization(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class SelfAttn(nn.Module):
  def __init__(self, ch, normalization=Normalization):
    super(SelfAttn, self).__init__()
    self.ch = ch
    
    self.norm = normalization(ch)
    
    # Key
    self.query = nn.Conv1d(self.ch, self.ch,#//8,
                            1, bias = False)
    self.key = nn.Conv1d(self.ch, self.ch,#//8,
                            1, bias = False)
    self.value = nn.Conv1d(self.ch, self.ch,#//2,
                            1, bias=False)
    self.out = nn.Conv1d(self.ch,#//2,
                            self.ch, 1, bias=False)
    
    # Gain parameter
    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

  def forward(self, x):
    h = x
    #h = self.norm(h)
  
    # query
    q = self.query(h)
    # key
    #k = F.max_pool1d(self.key(h), [2])
    k = self.key(h)
    # value
    #v = F.max_pool1d(self.value(h), [2])
    v = self.value(h)

    b, c, l = q.shape
    # Matmul and softmax to get attention maps
    w = torch.bmm(q.transpose(1,2), k)
    w = w * (c ** (-0.5))
    w = F.softmax(w, dim=-1)
    # Attention map times g path
    out = self.out(torch.bmm(v, w.transpose(1,2)))

    #return out + x
    return self.gamma * out + x

######https://github.com/lucidrains/perceiver-pytorch
''' 
class SinusoidalEmbeddings(nn.Module):
  def __init__(self,dim):
    super().__init__()
    inv_freq = 1./(10000**(torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)

  def forward(self, x):
    n = x.shape[-2]
    t = torch.arange(n, device = x.device).type_as(self.inv_freq)
    sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
    emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

    return emb[None, :, :]

def rotate_every_two(x):
  x = rearrange(x, '... (d j) -> ... d j', j=2)
  x1, x2 = x.unbind(dim = -1)
  x = torch.stack((-x2, x1), dim=-1)
  return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_emb(q, k, sinu_pos):
  sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
  sin, cos = sinu_pos.unbind(dim = -2)
  sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
  q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
  return q, k

def exists(val):
  return val is not None

def default(val, d):
  return val if exists(val) else d

def cache_fn(f):
  cache = None
  @wraps(f)
  def cached_fn(*args, _cache = True, **kwargs):
    if not _cache:
      return f(*args, **kwargs)
    nonlocal cache
    if cache is not None:
      return cache
    cache = f(*args, **kwargs)
    return cache
  return cached_fn

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
  x = x.unsqueeze(-1)
  device, dtype, orig_x = x.device, x.dtype, x

  scales = torch.logspace(0., log(max_freq / 2)/ log(base), num_bands, base = base, device= device, dtype=dtype)
  scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

  x = x * scales * pi
  x = torch.cat([x.sin(), x.cos()], dim=-1)
  x = torch.cat((x, orig_x), dim =-1)
  return x 


class PreNorm(nn.Module):
  def __init__(self, dim, fn, context_dim = None):
    super().__init__()
    self.fn = fn
    self.norm = nn.LayerNorm(dim)
    self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

  def forward(self, x, **kwargs):
      x = self.norm(x)
      
      if exists(self.norm_context):
        context = kwargs['context']
        normed_context = self.norm_context(context)
        kwargs.update(context = normed_context)

      return self.fn(x, **kwargs)

class GEGLU(nn.Module):
  def forward(self, x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)

class FeedForward(nn.Module):
  def __init__(self, dim, mult = 4, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2),
                            GEGLU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * mult, dim))
  
  def forward(self, x):
    return self.net(x)

class Attention(nn.Module):
  def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
    super().__init__()
    inner_dim = dim_head * heads
    context_dim = default(context_dim, query_dim)

    self.scale = dim_head ** -0.5
    self.heads = heads

    self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

    self.to_out = nn.Sequential(
      nn.Linear(inner_dim, query_dim),
      nn.Dropout(dropout)
    )

  def forward(self, x, context = None, mask = None, pos_emb = None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k, v = self.to_kv(context).chunk(2, dim = -1)

    # rearrange dims (batch, len, heads * dim_head) -> (batch * heads, len, dim_head) 
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

    if exists(pos_emb):
      q, k = apply_rotary_emb(q, k, pos_emb)

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
      mask = rearrange(mask, 'b, ... -> b (...)')
      max_neg_value = -torch.finfo(sim.dtype).max
      mask = repeat(mask, 'b j -> (b h) () j', h = h)
      sim.masked_fill(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
    return self.to_out(out)

class AttnBlock(nn.Module):
  def __init__(self,
    *, 
    num_freq_bands,
    depth,
    max_freq,
    freq_base = 2,
    input_channels = 2,
    input_axis = 2,
    num_latents = 512,
    latent_dim = 512,
    cross_heads = 1,
    latent_heads = 8, 
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,
    attn_dropout = 0., 
    ff_dropout = 0.,
    weight_tie_layers = False,
    fourier_encode_data = True, 
    self_per_cross_attn = 1,
    self_attn_rel_pos = True
  ):
    super().__init__()
    self.input_axis = input_axis
    self.max_freq = max_freq
    self.num_freq_bands = num_freq_bands
    self.freq_base = freq_base

    self.fourier_encode_data = fourier_encode_data
    fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
    input_dim = fourier_channels + input_channels

    #self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
    
    get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads,
                             dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
    get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
    get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads,
                                     dim_head = latent_dim_head, dropout = attn_dropout))
    get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

    get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

    self.layers = nn.ModuleList([])
    for i in range(depth):
      should_cache = i > 0 and weight_tie_layers
      cache_args = {'_cache': should_cache}

      self_attns = nn.ModuleList([])

      for _ in range(self_per_cross_attn):
        self_attns.append(nn.ModuleList([
          get_latent_attn(**cache_args),
          get_latent_ff(**cache_args)
        ]))

      self.layers.append(nn.ModuleList([
        get_cross_attn(**cache_args),
        get_cross_ff(**cache_args),
        self_attns
      ]))

    self.to_logits = nn.Sequential(
      nn.LayerNorm(latent_dim),
      nn.Linear(latent_dim, num_classes)
    )

    self.sinu_emb = None
    if self_attn_rel_pos:
      self.sinu_emb = SinusoidalEmbeddings(latent_dim_head)

  def forward(self, latents, data, mask = None):
    b, *axis, _, device = *data.shape, data.device
    assert len(axis) == self.input_axis

    if self.fourier_encode_data:
      # calculate fourier encoded positions in the range of [-1, 1], for all axis

      axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
      pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
      enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.freq_base)
      enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
      enc_pos = repeat(enc_pos, '... -> b ...', b = b)

      data = torch.cat((data, enc_pos), dim = -1)

    # concat to channels of data and flatten axis

    data = rearrange(data, 'b ... d -> b (...) d')

    #x = repeat(self.latents, 'n d -> b n d', b = b)
    x = latents

    # rotary embeddings for latents, if specified

    pos_emb = self.sinu_emb(x) if exists(self.sinu_emb) else None

    # layers

    for cross_attn, cross_ff, self_attns in self.layers:
      x = cross_attn(x, context = data, mask = mask) + x
      x = cross_ff(x) + x

      for self_attn, self_ff, in self_attns:
        x = self_attn(x, pos_emb = pos_emb) + x
        x = self_ff(x) + x
  
    return x
'''
##############################TransGan

def gelu(x):
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Mlp(nn.Module):
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(drop)

  def forward(self, x):
    x = self.fc1(x)
    X = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x

def get_attn_mask(N, w):
  mask = torch.zeros(1, 1, N, N).cuda()
  for i in range(N):
    if i <= w:
      mask[:, :, i, 0:i+w+1] = 1
    elif N - i <= w:
      mask[:, :, i, i - w: N] = 1
    else:
      mask[:, :, i, i: i+w+1] = 1
      mask[:, :, i, i-w: i] = 1
  return mask


class Attention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., is_mask=0):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    
    self.scale = qk_scale or head_dim // num_heads

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)
    self.is_mask = is_mask
    self.remove_mask = False
    self.mask_4 = get_attn_mask(is_mask, 8)
    self.mask_5 = get_attn_mask(is_mask, 10)
    self.mask_6 = get_attn_mask(is_mask, 12)
    self.mask_7 = get_attn_mask(is_mask, 14)
    self.mask_8 = get_attn_mask(is_mask, 16)
    self.mask_10 = get_attn_mask(is_mask, 20)

  def forward(self, x, epoch=0):
    # (batch, seq_len, dim)
    B, N, C = x.shape
    # (3(qkv), batch, heads, seq_len, dim//heads)
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # (batch, heads, seq_len(N), seq_len(N))
    attn = (torch.matmul(q, k.transpose(-1, -2))) * self.scale

    if self.is_mask:
      if epoch < 20:
        if epoch < 5:
          mask = self.mask_4
        elif epoch < 10:
          mask = self.mask_6
        if epoch < 15:
          mask = self.mask_8
        else:
          mask = self.mask_10
      attn = attn.masked_fill(mask.to(attn.get_device()) == 0, 1e-9)
    else:
      pass

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # (batch, heads, seq_len, dim // heads) -> (batch, seq_len, heads, dim // heads) -> (batch, seq_len, dim)
    x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


# Attention block (Attn + MLP)
class Block(nn.Module):
  def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
              drop_path=0., act_layer=gelu, norm_layer = nn.LayerNorm, is_mask=0):
    super().__init__()
    self.norm1 = norm_layer(dim)
    self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, is_mask=is_mask)
    
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(dim)

    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer = act_layer, drop=drop)

  def forward(self, x, epoch=0):
    x = x + self.drop_path(self.attn(self.norm1(x), epoch))
    x = x + self.drop_path(self.mlp(self.norm2(x)))

    return x

class SampleShuffle(nn.Module):
  def __init__(self, scale):
    super().__init__()
    self.scale = scale

  def forward(self, x):
    assert len(x.shape) == 3
    B, C, L = x.size()
    assert C % self.scale == 0

    nc = C//self.scale
    nl = L*self.scale

    x = x.reshape(B, nc, self.scale, L).transpose(-1, -2)
    x = x.reshape(B, nc, nl)
    return x

class SampleUnshuffle(nn.Module):
  def __init__(self, scale):
    super().__init__()
    self.scale = scale

  def forward(self, x):
    assert len(x.shape) == 3
    B, C, L = x.size()
    assert L % self.scale == 0

    nc = C*self.scale
    nl = L//self.scale

    x = x.reshape(B, C, nl, self.scale).transpose(-1, -2)
    x = x.reshape(B, nc, nl)
    return x


def audio_upsample(x):
  B, L, C = x.size()

  x = x.permute(0, 2, 1)
  x = SampleShuffle(4)(x)
  B, C, L = x.size()
  x = x.permute(0, 2, 1)

  return x, L


def audio_downsample(x):
  B, L, C = x.size()

  x = x.permute(0, 2, 1)
  x = SampleUnshuffle(4)(x)
  B, C, L = x.size()
  x = x.permute(0, 2, 1)
  
  return x, L

if __name__ == '__main__':
  test = torch.randn(10, 10, 10)
  test = torch.Tensor([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]])
  shuffle = SampleShuffle(2)(test)
  unshuffle = SampleUnshuffle(2)(shuffle)
  print(test, shuffle, unshuffle)
  assert torch.all(test == unshuffle)


















