import torch
import torch.nn as nn
from models.vq_vae.modules import VQVAEEncoder, VQVAEDecoder, VectorQuantizer

class VQVAE(nn.Module):
  def __init__(self, embed_dim, n_embed, in_ch, out_ch, num_chs, depth, attn_indices):
    super(VQVAE, self).__init__()
    self.embed_dim = embed_dim
    self.n_embed = n_embed

    self.in_ch = in_ch
    self.out_ch=out_ch
  
    enc_attn_indices = attn_indices
    dec_attn_indices = [(len(num_chs)-1)-i for i in attn_indices]
   
    self.encoder = VQVAEEncoder(in_ch, num_chs[-1], num_chs, depth, enc_attn_indices)
    # conv that changes filter number to vq dimension
    self.enc_conv = nn.Conv1d(num_chs[-1], embed_dim, 3, padding=1)

    self.vector_quantizer = VectorQuantizer(embed_dim, n_embed)
    
    self.dec_conv = nn.Conv1d(embed_dim, num_chs[-1], 3, padding=1)
    self.decoder = VQVAEDecoder(num_chs[-1], out_ch, num_chs[::-1], depth, dec_attn_indices)

    self.tanh = nn.Tanh()
    
  # returns os the vectors zq, ze and the indices
  def encode(self, inputs):
    #inputs_one_hot = F.one_hot(inputs, self.in_ch).permute(0, 2, 1).float()

    encoding = self.encoder(inputs)
    
    encoding = self.enc_conv(encoding)
    quant, codes, indices = self.vector_quantizer(encoding.permute(0, 2, 1))
    quant = quant.permute(0, 2, 1)

    return encoding, quant, codes, indices

  def decode(self, quant):
    reconstructed = self.dec_conv(quant)
    reconstructed = self.decoder(reconstructed)
    reconstructed = self.tanh(reconstructed)

    return reconstructed

  # a way to get the codebook
  def get_vq_vae_codebooks(self):
    codebook = self.vector_quantizer.quantize(np.arange(self.n_embed))
    codebook = codebook.reshape(self.n_embed, self.embed_dim)

    return codebook

  def get_last_layer(self):
    return self.decoder.last_conv.weight

  def forward(self, inputs):
    encoding, quant, codes, indices = self.encode(inputs)

    reconstructed = self.decode(quant)    

    return reconstructed, codes
