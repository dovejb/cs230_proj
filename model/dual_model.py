import sys
sys.path.append("..")
from torch import nn, Tensor
import torch
from .model import Model
import metrics

class DualModel(nn.Module):
    def __init__(self, 
                wave_length=8000,
                window_size=512,
                hop_size=160,
                num_features=512,
                num_heads=8,
                local_size=5,
                dropout=0.2,
                tcn_bottleneck_dim=128,
                num_tcn_block_channels=1024,
                num_dilations=4,
                num_repeats=2,
                ):
        super(DualModel, self).__init__()
        self.fd = Model(
            1,
            wave_length,
            window_size,
            hop_size,
            num_features,
            num_heads,
            local_size,
            dropout,
            tcn_bottleneck_dim,
            num_tcn_block_channels,
            num_dilations,
            num_repeats,
        )
        self.td = Model(
            2,
            wave_length,
            window_size,
            hop_size,
            num_features,
            num_heads,
            local_size,
            dropout,
            tcn_bottleneck_dim,
            num_tcn_block_channels,
            num_dilations,
            num_repeats,
        )
    def forward(self, input:Tensor):
        fout = self.fd(input) 
        tout = self.td(input)
        output = (fout + tout)/2
        return output
