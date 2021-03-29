import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
  def __init__(self, embedding_dim, num_embeddings):
    """
    Args:
      embedding_dim: dimensionality of the tensors in the quantized space.
        Inputs to the modules must be in this format as well.
      num_embeddings: number of vectors in the quantized space.
    """
    super(VectorQuantizer, self).__init__()
    # k: size of the discrete latent space
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    # d: dimensionalidof each embedding latent vector
    # (k embedding vectors)
    # codebook: contains the k d-dimensional vectors from the quantized latent space
    emb = torch.empty(embedding_dim, num_embeddings)
    emb.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    #torch.nn.init.xavier_uniform_(emb)
    self.embedding= nn.Parameter(emb)
    self.register_parameter('embeddings', self.embedding)  

    #embeddings = nn.Embedding(num_embeddings, embedding_dim)
  
  def forward(self, inputs):
    """Connects the module to some inputs.
    Args:
      inputs: Tensor, second dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.
   
   Returns:
        quantize: Tensor containing the quantized version of the input.
        encodings: Tensor containing the discrete encodings, ie which element
        of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
    """
    # input (batch, len, dim) -> (batch*len, dim)    
    flat_inputs = inputs.reshape(-1, self.embedding_dim)
    # distance between the input and the embedding elements
    distances = (
        torch.sum(flat_inputs**2, 1, keepdim=True)
        - 2*torch.matmul(flat_inputs, self.embeddings)
         + torch.sum(self.embeddings**2, 0, keepdim=True)
    )
    
    # index with smaller distance beetween the input and the embedding elements
    encoding_indices = torch.argmax(-distances, dim=1)
    # transform the index in one_hot to multiply by the embedding
    encodings = nn.functional.one_hot(encoding_indices, self.num_embeddings).type_as(distances)
    # multiply the index to find the quantization
    encoding_indices = encoding_indices.view(*inputs.shape[:-1])
    #print(set(encoding_indices.flatten().tolist()))
    quantized = self.quantize(encoding_indices)
    
    # before the detach
    codes = torch.cat([inputs, quantized], axis=-1)

    # straight through estimator
    quantized = inputs + (quantized - inputs).detach()
    avg_probs = torch.mean(encodings, 0)
    
    return quantized, codes, encoding_indices
  
  def quantize(self, encoding_indices):
    """Retorna vetor de embedding para um batch de índices"""
    return nn.functional.embedding(encoding_indices, self.embeddings.transpose(1,0))

class ResLayer(nn.Module):
  def __init__(self, filters, dilation, leaky=False):
    super(ResLayer, self).__init__()
    padding = dilation
    self.conv = nn.Sequential(
                  nn.Conv1d(filters, filters, 3, dilation=dilation, padding=padding),
                  nn.BatchNorm1d(filters),
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  nn.Conv1d(filters, filters, 3, padding=1),
                  nn.BatchNorm1d(filters))
    self.relu =  nn.LeakyReLU(0.2) if leaky else nn.ReLU()
       
  def forward(self, inputs):
    res_x = self.conv(inputs)
    res_x += inputs
    res_x = self.relu(res_x)
    return res_x

class ResBlock(nn.Module):
  def __init__(self, filters, dilations, depth, leaky=False):
    super(ResBlock, self).__init__()
    self.res_block = nn.Sequential(*[ResLayer(filters, dilations[i], leaky) for i in range(depth)])

  def forward(self, inputs):
    output = self.res_block(inputs)
    return output
  
  class VQVAEEncoder(nn.Module):
  def __init__(self, first_filter, last_filter, num_filters, depth, leaky=False):
    super(VQVAEEncoder,self).__init__()
    dilations = [3**i for i in range(depth)]

    # possível camada para condicionamento
    # self.cond = nn.AdaptiveMaxpool()

    kernel_size = 3
    stride=4
    padding = (kernel_size-stride)//2

    self.causal_conv = nn.Conv1d(first_filter, num_filters[0], 3, padding=1)

    res_list = []
    for i in range(len(num_filters)):
      res_list.extend([ResBlock(num_filters[i], dilations, depth, leaky),
                      nn.Conv1d(num_filters[i], num_filters[i], kernel_size, stride=stride, padding=padding),
                      nn.BatchNorm1d(num_filters[i]),
                      nn.LeakyReLU(0.2) if leaky else nn.ReLU()
                      ])
    self.res_layers =  nn.Sequential(*res_list)

    # post processing layers
    self.post = nn.Sequential(nn.Conv1d(num_filters[-1], num_filters[-1], 3, padding=1),
                              nn.BatchNorm1d(num_filters[-1]),
                              nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                              nn.Conv1d(num_filters[-1], last_filter, 3, padding=1))
    
  def forward(self, inputs):
    x = self.causal_conv(inputs)
    x = self.res_layers(x)
    output = self.post(x)
    return output

class VQVAEDecoder(nn.Module):
  def __init__(self, first_filter, last_filter, num_filters, depth, leaky=False):
    super(VQVAEDecoder,self).__init__()
    dilations = [3**i for i in range(depth)]
    dilations = dilations[::-1]
    self.causal_conv = nn.Conv1d(first_filter, num_filters[0], 3, padding=1)

    kernel_size =3
    stride = 4
    output_padding = 0
    padding = (kernel_size + output_padding-stride)//2

    res_list = []
    for i in range(len(num_filters)):
      res_list.extend([ResBlock(num_filters[i], dilations, depth, leaky),
                      nn.ConvTranspose1d(num_filters[i], num_filters[i], kernel_size, stride=stride, padding=padding),
                      nn.BatchNorm1d(num_filters[i]),
                      nn.LeakyReLU(0.2) if leaky else nn.ReLU()])
    self.res_layers =  nn.Sequential(*res_list)

    # post processing layers
    self.post = nn.Sequential(
                          nn.Conv1d(num_filters[-1], num_filters[-1], 3, padding=1),
                          nn.BatchNorm1d(num_filters[-1]),
                          nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                          nn.Conv1d(num_filters[-1], last_filter, 3, padding=1))
    
  def forward(self, inputs):
    x = self.causal_conv(inputs)
    x = self.res_layers(x)
    output = self.post(x)
    return output

class SelfAttn(nn.Module):
  def __init__(self, ch, activation):
    super(SelfAttn, self).__init__()
    self.ch = ch
    self.activation = activation

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

  
  
