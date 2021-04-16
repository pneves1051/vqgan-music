import torch
import torch.nn as nn
import torch.nn.functional as F
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