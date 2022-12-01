from torch import nn, Tensor
import torch
from torch.nn.utils import weight_norm
from .embedding import *

"""
    Temporal Convolutional Network for Mask Estimation
"""

class DepthwiseConv1D(nn.Module):
    def __init__(self, in_chan=512, T=50, out_chan=1024, kernel=3, dilation=1):
        super(DepthwiseConv1D, self).__init__()

        padding1 = (kernel-1)//2*dilation
        self.conv1 = nn.Conv1d(
            in_channels=in_chan,
            out_channels=in_chan,
            kernel_size=kernel,
            stride=1,
            dilation=dilation,
            padding=padding1,
        )
        self.prelu = nn.PReLU()
        self.layernorm = nn.LayerNorm((in_chan,T))
        self.conv2 = nn.Conv1d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input:Tensor):
        x = self.conv1(input)
        x = self.prelu(x)
        x = self.layernorm(x)
        output = self.conv2(x)
        return output

class TemporalBlock(nn.Module):
    def __init__(self, T, chan=512, block_chan=1024, dilation=1):
        super(TemporalBlock, self).__init__()
        self.T = T
        self.chan = chan

        self.pointconv = nn.Conv1d(
            in_channels=chan,
            out_channels=block_chan,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.prelu = nn.PReLU(block_chan)
        self.layernorm = nn.LayerNorm((block_chan, T))
        self.main = DepthwiseConv1D(block_chan, T, chan, 3, dilation)

    def forward(self, input:Tensor):
        x = input
        assert x.shape[-1] == self.T and x.shape[-2] == self.chan
        x = self.pointconv(x)
        x = self.prelu(x)
        x = self.layernorm(x)
        x = self.main(x)
        output = x + input
        return output

class TCN(nn.Module):
    def __init__(self, T, chan=1, bottleneck_dim=128, block_chan=64, dropout=0.2, dilation_n=4, repeat_n=2):
        super(TCN, self).__init__()
        self.chan = chan
        self.T = T
        self.bottleneck_dim = bottleneck_dim
        self.repeat_n = repeat_n
        self.dilation_n = dilation_n

        self.layernorm = nn.LayerNorm(T)
        self.bottleneck = nn.Conv1d(
            in_channels=chan,
            out_channels=bottleneck_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.blocks = nn.ModuleList([])
        for i in range(repeat_n):
            dilation = 1
            for j in range(dilation_n):
                self.blocks.append(TemporalBlock(
                    T,
                    chan=bottleneck_dim,
                    block_chan=block_chan,
                    dilation=dilation, 
                ))
                dilation *= 2
        self.prelu = nn.PReLU(bottleneck_dim)
        self.separate = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=chan,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.activation = nn.ReLU()
    
    def forward(self, input:Tensor):
        """
            input should be [N,F,T]
            output is mask [N,F,T]

            (block input should be [N,F,T])
        """
        x = input
        assert x.shape[-1] == self.T and x.shape[-2] == self.chan, f"shape mismatch: x is {x.shape[1:]}, require {(self.chan,self.T)}"
        x = self.layernorm(x)
        x = self.bottleneck(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.prelu(x)
        x = self.separate(x)
        output = self.activation(x)
        return output

if __name__ == '__main__':
    model = TCN(T=50,chan=512)
    from torchinfo import summary
    summary(model, (16,50,512))
