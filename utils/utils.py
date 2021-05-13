import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings
from collections import defaultdict

# initialization
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

def encode_dataset(dataloader, vqvae, device):
  encoded_dataset = {'ids': [], 'inputs': [], 'conditions':[]}
  
  vqvae.eval()
  with torch.no_grad():
    for j, data in enumerate(dataloader):
      if j % 500 == 0 : print(len(encoded_dataset['inputs']))
      
      ids = data['ids'].tolist()
      real = data['inputs'].to(device)
      conditions = data['conditions']
      if conditions is not None:
        conditions = conditions.to(device)
      
      _, _, _, indices = vqvae.encode(real)
      
      for i, id in enumerate(ids):

        if id in encoded_dataset['ids']:
          
          pos_id = encoded_dataset['ids'].index(id)
          encoded_dataset['inputs'][pos_id].extend(indices[i].cpu().tolist())
          '''
          if conditions is not None:
            encoded_dataset['conditions'][pos_id].extend(conditions.cpu().tolist())
          else:
            encoded_dataset['conditions'][pos_id].extend([None])
          '''
        else:
          encoded_dataset['ids'].append(id)
          encoded_dataset['inputs'].append(indices[i].cpu().tolist())
          if conditions is not None:
            encoded_dataset['conditions'].append(conditions[i].cpu().tolist())
          else:
            encoded_dataset['conditions'].append([None])
   
  return encoded_dataset


def generate(input, conditions, vqvae, transformer, generate, past, contraction, temperature, device):
  #x = encode_mu_law(input)[np.newaxis, ..., np.newaxis]
  vqvae.eval()
  transformer.eval()
  
  with torch.no_grad():
    generated = []

    _,_,_,tr_input = vqvae.encode(input)

    input_size = tr_input.shape[-1]
        
    total_data = tr_input.clone()
        
    generate = generate//contraction
    past = past//contraction

    tr_input = tr_input[:, -past:]
    
    for i in range(generate):
      if i% 100 == 0: print(i)
      mask = transformer.generate_square_subsequent_mask(tr_input.size(1)).to(device)
      
      # predictions.shape = (batch_size, vocab_size, seq_len)
      predictions = transformer(tr_input, mask)
      
      print(predictions.shape)
      # selects the last output in the seq_len dimension
      predictions= predictions[:, :, -1] # (1, vocab_size)

      predictions /= temperature   
      predicted_id = torch.distributions.Categorical(F.softmax(predictions, dim=1)).sample()
      #top_predicted_id = torch.argmax(F.softmax(top_predictions, dim=1), dim=1)
      if i % 8 == 0: print(predicted_id)
      # concatenated predicted_id to the output, which is given to the decoder as input
      total_data = torch.cat([total_data, predicted_id.unsqueeze(-1)], axis=-1)

      tr_input =  total_data[:, -past:]
          
      #print(bottom_tr_input.shape)

    # index prediction
    #generated = tf.expand_dims(generated,0)
    total_data = vqvae.vector_quantizer.quantize(total_data).transpose(1,2)
    
    print(total_data.shape) 
    generated = vqvae.decode(total_data)
    generated = generated[0]
    # decoding 
    #generated = decode_mu_law(np.array(generated))
    return generated


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)