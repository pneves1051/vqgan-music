# Patch based multiscale discriminator
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq_vae.attention import Block, audio_upsample, SelfAttn
from utils.utils import trunc_normal_


class WNConv1d(nn.Module):
  def __init__(self, *args, **kwargs):
    super(WNConv1d, self).__init__()
    self.conv = nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))

  def forward(self, x):
    return self.conv(x)


class ModuleDiscriminator(nn.Module):
  def __init__(self, in_ch, num_chs, stride, window_size, cont):
    super(ModuleDiscriminator, self).__init__()
    shuffle_n = 0
 
    self.pre = nn.Sequential(WNConv1d(in_ch, num_chs[0], 9, 
                                      stride=1, padding=4),
                              nn.LeakyReLU(0.2))
    
    module_list = []
    for i in range(1, len(num_chs)):
      module_list.append(nn.Sequential(WNConv1d(num_chs[i-1], num_chs[i],
                                                                      kernel_size=stride * 10 + 1,
                                                                      stride=stride,
                                                                      padding=stride * 5,
                                                                      groups=num_chs[i-1] // 4),                       
                        nn.LeakyReLU(0.2)))
    
    module_list.append(WNConv1d(num_chs[-1], 1, kernel_size=3, stride=1, padding=1))
    
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


###################ATTN(Trnasgan)

class PatchEmbed(nn.Module):
  """ Audio to patch embedding
  """
  def __init__(self, sample_length = 65536, patch_size = 256, in_channels=1, embed_dim = 768):
    super().__init__()
    num_patches = sample_length//patch_size
    self.sample_length = sample_length
    self.patch_size = patch_size
    self.num_patches = num_patches

    self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

  def forward(self, x):
    B, C, L = x.shape
    assert L == self.sample_length, \
      f"Input sample_length ({L}) doesn't match model ({self.sample_length})."
    
    # (batch, chan, length) -> (batch, length, chan)
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x


class HybridEmbed(nn.Module):
  """ CNN Feature Map Embedding
  Extract feature map from CNN, flatten, project to embedding dim
  (hybrid because cnn + linear)
  """
  def __init__(self, backbone, sample_length=65536, feature_size=None, in_channels=1, embed_dim=768):
      super().__init__()
      assert isinstance(backbone, nn.Module)
      self.img_size = sample_length
      self.backbone = backbone
      if self.feature_size is None:
        with torch.no_grad():
          training = backbone.training
          if training:
            backbone.eval()
          o = self.backbone(torch.zeros(1, in_channels, sample_length))[-1]
          feature_size = o.shape[-1]
          feature_dim = o.shape[1]
          backbone.train(training)
      else: 
        feature_dim = self.backbone.feature_info.channels()[-1]
      self.num_patches = feature_size
      self.proj = nn.Linear(feature_dim, embed_dim)
  def forward(self, x):
    x = self.backbone(x)[-1]
    x = x.flatten(2).transpose(1, 2)
    x = self.proj(x)
    return x


class AttnDiscriminator(nn.Module):
  def __init__(self, embed_dim=384, sample_length=65536, patch_size=256, in_channels=1, num_classes=10, depth=7,
                  num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  drop_path_rate=0., hybtrid_backbone=None, norm_layer=nn.LayerNorm):
    super(AttnDiscriminator, self).__init__()
    self.num_classes = num_classes
    self.num_features = embed_dim
    patch_size = patch_size

    if hybtrid_backbone is not None:
      self.patch_embed = HybridEmbed(
        hybtrid_backbone, sample_length=sample_length, in_channels=in_channels, embed_dim=embed_dim)
    else:
      self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
    num_patches = sample_length // patch_size

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule
    self.blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)])

    self.norm = norm_layer(embed_dim)

    self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    trunc_normal_(self.pos_embed, std=.02)
    trunc_normal_(self.cls_token, std=.02)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  @torch.jit.ignore
  def no_weight_decay(self):
    return{'pos_embed', 'cls_token'}

  def get_classifier_head(self):
    return self.head

  def reset_classifier(self, num_classes, global_pool=''):
    self.num_classes = num_classes
    self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

  def forward_features(self, x):
    B = x.shape[0]
    # (batch, chan, len) -> (batch, len, chan)
    x = self.patch_embed(x).flatten(2).permute(0, 2, 1)

    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)
    for block in self.blocks:
      x = block(x)

    x = self.norm(x)
    return x[:, 0]

  def forward(self, x):
    results = []
    x = self.forward_features(x)
    results.append(x)
    x = self.head(x)
    results.append(x)
    
    return (results,)
