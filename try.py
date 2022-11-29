import torch.nn.functional as F
from torch import nn
import torch

if __name__ == '__main__':
    x = torch.randn((16,8,50))
    x = x.view((x.shape[0], x.shape[1]//2, 2, x.shape[2]))
    print(x.shape)
