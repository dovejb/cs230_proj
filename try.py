import torch.nn.functional as F
from torch import nn
import torch

if __name__ == '__main__':
    wnd = torch.hann_window(20)
    print(wnd)
    wnd = (wnd + torch.ones((20,))) / 2
    print(wnd)