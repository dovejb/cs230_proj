import torch.nn.functional as F
import torch

x = torch.Tensor([0.5,0.5])
print(F.l1_loss(x, x-x))
