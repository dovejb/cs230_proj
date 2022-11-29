from torch import nn, Tensor
import torch
from torch.nn.utils import weight_norm
from embedding import *

"""
    Temporal Convolutional Network for Mask Estimation
"""

class DepthwiseConv1D(nn.Module):
    def __init__(self, in_chan=1, out_chan=3, kernel=3, dilation=1):
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
        self.conv2 = nn.Conv1d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input:Tensor):
        x = self.conv1(input)
        output = self.conv2(x)
        return output

class TemporalBlock(nn.Module):
    def __init__(self, T, chan=1, block_chan=64, dilation=1):
        super(TemporalBlock, self).__init__()
        self.T = T
        self.chan = chan

        self.extend = nn.Conv1d(
            in_channels=chan,
            out_channels=block_chan,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.layernorm1 = nn.LayerNorm((block_chan, T))
        self.prelu1 = nn.PReLU(block_chan)
        self.main = DepthwiseConv1D(block_chan, chan, 3, dilation)
        self.layernorm2 = nn.LayerNorm((chan, T))
        self.prelu2 = nn.PReLU(chan)

    def forward(self, input:Tensor):
        x = input
        assert x.shape[-1] == self.T and x.shape[-2] == self.chan
        x = self.extend(x)
        x = self.layernorm1(x)
        x = self.prelu1(x)
        x = self.main(x)
        x = self.layernorm2(x)
        x = self.prelu2(x)
        return x + input

class TCN(nn.Module):
    def __init__(self, T, chan=1, block_chan=64, dropout=0.2, dilation_n=4, repeat_n=2):
        super(TCN, self).__init__()
        self.chan = chan
        self.T = T
        self.repeat_n = repeat_n
        self.dilation_n = dilation_n

        self.blocks = nn.ModuleList([])
        for i in range(repeat_n):
            dilation = 1
            for j in range(dilation_n):
                self.blocks.append(TemporalBlock(
                    T,
                    chan=chan,
                    block_chan=block_chan,
                    dilation=dilation, 
                ))
                dilation *= 2
        self.activation = nn.Sigmoid()
    
    def forward(self, input:Tensor):
        """
            input should be (after Att Encoder) [N, T, F] (F is num features)
            output is mask [N,F,T]

            (block input should be [N,F,T])
        """
        x = input
        assert x.shape[-2] == self.T and x.shape[-1] == self.chan, f"shape mismatch: x is {x.shape[1:]}, require {(self.T,self.chan)}"
        x = x.permute(0,2,1)
        for block in self.blocks:
            x = block(x)
        output = self.activation(x)
        return output

if __name__ == '__main__':
    model = TCN(T=50,chan=512)
    from torchinfo import summary
    summary(model, (16,50,512))
