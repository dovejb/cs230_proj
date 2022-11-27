import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchinfo import summary
import numpy as np
from constants import *
import evaluate

class ResBlock(nn.Module):
    def __init__(self,
                    n_chan,
                    kernel=3,
                    stride=1,
                    #padding=1,
                    loop_count=1,
                    ):
        super(ResBlock, self).__init__()
        self.n_chan = n_chan 
        self.kernel = kernel
        self.stride = stride
        self.padding = (kernel-1)//2 #padding
        self.loop_count = loop_count

        self.conv = nn.Conv1d(
                            in_channels=self.n_chan,
                            out_channels=self.n_chan,
                            kernel_size=self.kernel,
                            stride=self.stride,
                            padding=self.padding,
                        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.activate = self.tanh
        self.batchnorm = nn.BatchNorm1d(self.n_chan)

    def forward(self, input=Tensor):
        x = input
        for i in range(self.loop_count):
            x = self.conv(x)
            x = self.batchnorm(x)
            x = self.activate(x)
        x = self.conv(x)
        #x = self.batchnorm(x)

        add = x + input
        output = self.activate(add)
        return output
        

class WaveCNN(nn.Module):
    def __init__(self,
                    wave_length=16000,  # SR:16000 Time:1s
                    window_size=8,    
                    block_num=3,
                    res_loop_count=3,
                    initial_n_chan=512,
                    res_kernel=3,
                    ):
        super(WaveCNN, self).__init__()

        self.wave_length=wave_length
        self.window_size=window_size
        self.block_num = block_num
        self.res_loop_count = res_loop_count
        self.res_kernel = res_kernel

        n_chan = initial_n_chan 
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=n_chan,
            kernel_size=self.window_size,
            stride=self.window_size//2,
            padding=self.window_size//2,
        )

        self.resblocks = nn.ModuleList([])
        self.pointwises = nn.ModuleList([])
        for i in range(self.block_num):
            n_half = n_chan//2
            self.resblocks.append(
                ResBlock(n_chan, kernel=self.res_kernel, loop_count=self.res_loop_count),
            )
            self.pointwises.append(
                nn.Conv1d(n_chan,n_half,1)
            )
            n_chan = n_half
        
        self.convtranspos1d = nn.ConvTranspose1d(
            in_channels=n_chan,
            out_channels=1,
            kernel_size=self.window_size,
            stride=self.window_size//2,
            padding=self.window_size//2,
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input:Tensor):
        assert input.shape[-2] == 1 and input.shape[-1] == self.wave_length
        x = input 

        # Downsample
        x = self.conv1d(x)

        # Res Blocks
        for i in range(self.block_num):
            x = self.resblocks[i](x)
            x = self.pointwises[i](x)
            #x = self.relu(x)
            x = self.tanh(x)
        
        # Upsample -> mask
        mul = self.convtranspos1d(x)
        mul = self.sigmoid(mul)
        add = self.convtranspos1d(x)
        add = self.sigmoid(add)

        # Output
        output = input * mul + add
        output = output - torch.mean(output)
        return output
    
    def loss_function(self,
                      y:Tensor,
                      yhat:Tensor,
                      *args,
                      **kw) -> dict:
        results = []
        x = kw['x']
        sisnr = torch.sum(evaluate.sisnr(y, yhat))
        sisnrX = torch.sum(evaluate.sisnr(x, yhat))
        sisnri = torch.sum(sisnr - evaluate.sisnr(y, x))
        sdr = torch.sum(evaluate.new_sdr(y, yhat))
        mse = F.mse_loss(yhat, y)
        mean = torch.mean(yhat)-torch.mean(y)
        std = torch.std(yhat)-torch.std(y)
        results.append(("loss",-sisnr))#-0.2*sdr))
        results.append(("sisnr",sisnr))
        results.append(("sisnri",sisnri))
        results.append(("sdr",sdr))
        results.append(("mse",mse))
        results.append(("mean",mean))
        results.append(("std",std))
        results.append(("sisnrX",sisnrX))

        return results

    def normalized_y(self, mean, std):
        self.ynorm_mean = mean
        self.ynorm_std = std
    def denormalize_y(self, y):
        if self.ynorm_mean is None or self.ynorm_std is None:
            return y
        return 
    
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

if __name__ == '__main__':
    model = WaveCNN()
    summary(model, input_size=(32,1,16000)) 