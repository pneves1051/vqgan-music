import torch
import torch.nn as nn
from modules import VQVAEEncoder, VQVAEDecoder, VectorQuantizer


class VQVAE(nn.Module):
  def __init__(self, embedding_dim, num_embeddings, input_channels, output_channels, num_filters, depth):
    super(VQVAE, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings

    self.input_channels = input_channels
    self.output_channels=output_channels
  
    # o encoder bottom quantiza até um certo nível, e depois o encoder top termina a quantização
    self.encoder = VQVAEEncoder(input_channels, num_filters[-1], num_filters, depth)
    # esta convolução passará o número de filtros para a dimensão do vector quant.
    self.conv = nn.Conv1d(bottom_num_filters[-1], embedding_dim, 3, padding=1)

    self.vector_quantizer = VectorQuantizer(embedding_dim, num_embeddings)
    
    self.decoder = VQVAEDecoder(embedding_dim, output_channels, num_filters[::-1], depth)

    self.tanh = nn.Tanh()
    self.sigmoid= nn.Sigmoid()

  # retorna os vetores zq, ze e os índices a partor de inputs
  def encode(self, inputs):
    #inputs_one_hot = F.one_hot(inputs, self.input_channels).permute(0, 2, 1).float()

    encoding = self.encoder(inputs)
    
    encoding = self.conv(encoding)
    quant, codes, indices = self.vector_quantizer(encoding.permute(0, 2, 1))
    quant = quant.permute(0, 2, 1)

    return encoding, quant, codes, indices

  def decode(self, top_quant, bottom_quant):
    reconstructed = self.decoder(quant)
    reconstructed = self.tanh(reconstructed)

    return reconstructed

  #maneira de conseguirmos facilmente os codebooks utilizados
  def get_vq_vae_codebooks(self):
    codebook = self.vector_quantizer.quantize(np.arange(self.num_embeddings))
    codebook = codebook.reshape(self.num_embedings, self.embedding_dim)

    return top_codebook, bottom_codebook

  def forward(self, inputs):
    encoding, quant, codes, indices = self.encode(inputs)

    reconstructed = self.decode(quant, quant)    

    return reconstructed, codes
