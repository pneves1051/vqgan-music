import torch 
import torch.nn as nn

from models.transformers.transformer import Transformer


fake_data = torch.arange(0, 128).unsqueeze(0)
fake_target = torch.arange(1, 129).unsqueeze(0)
fake_cond = torch.arange(0,2).unsqueeze(0)

transformer = Transformer(
            vocab_size=256,
            embed_size=32,
            n_heads=8, 
            forward_expansion=4,
            n_layers=6, 
            max_len=8192, 
            dropout=0.1,
            nb_features=256,
            causal=True,
            cond='discrete',
            cond_dim=0)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr = 1e-4)

for i in range(100):
  out = transformer(fake_data, fake_cond)
  loss = loss_fn(out.transpose(-1,-2), fake_target)
  loss.backward()
  optimizer.step()

  print(loss.item())

test = transformer(fake_data)
print(test.shape)