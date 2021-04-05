import torch
from models.losses import multispectral_loss

test1 = torch.randn((1, 1, 10000))
test2 = torch.randn((1,1, 10000))

print(multispectral_loss(test1, test2))