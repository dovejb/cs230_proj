import sys
sys.path.append("..")
from torch import nn, Tensor
import torch
from .embedding import *
from .tcn import TCN
from .encoder import Encoder
from .decoder import Decoder
import metrics

def print_meta(name, x:Tensor):
    print(name, "mean", torch.mean(x), "std", torch.std(x))

class Model(nn.Module):
    def __init__(self, 
                embed_type=1,
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
        """
            embed_type: 1-STFT 2-CONV
        """
        super(Model, self).__init__()
        if embed_type == 1:
            self.embed_layer = EmbeddingSTFT(
                wave_length=wave_length,
                window_size=window_size,
                hop_size=hop_size,
            )
            self.output_layer = DecodeSTFT(self.embed_layer)
        elif embed_type == 2:
            self.embed_layer = EmbeddingConv(
                wave_length=wave_length,
                window_size=window_size,
                hop_size=hop_size,
                num_features=num_features,
            )
            self.output_layer = DecodeConv(self.embed_layer)
        else:
            raise Exception("Invalid embed_type")

        self.encoder_layer = Encoder(
            embed_dim=self.embed_layer.embedding_dim,
            T=self.embed_layer.T,
            num_heads=num_heads,
            dropout=dropout,
            local_size=local_size,
        )

        self.mask_layer = TCN(
            T=self.embed_layer.T,
            chan=self.embed_layer.embedding_dim,
            bottleneck_dim=tcn_bottleneck_dim,
            block_chan=num_tcn_block_channels,
            dropout=dropout,
            dilation_n=num_dilations,
            repeat_n=num_repeats,
        )

        self.decoder_layer = Decoder(
            embed_dim=self.embed_layer.embedding_dim,
            T=self.embed_layer.T,
            num_heads=num_heads,
            dropout=dropout,
            local_size=local_size,
        )

        self.loss_window = (torch.hann_window(wave_length) + torch.ones((wave_length,))) / 2
        self.loss_window = self.loss_window.cuda()
        self.layers = [self.embed_layer, self.encoder_layer, self.mask_layer, self.decoder_layer, self.output_layer]
    def freeze_layers(self, lno=[]):
        def freeze(layer):
            for param in layer.parameters():
                param.requires_grad = False
        for idx in lno:
            freeze(self.layers[idx])

    def forward(self, input_wave:Tensor):
        if input_wave.dim() == 1:
            input_wave = input_wave.view((1,input_wave.shape[-1]))
        # input_wave is [N,L]
        embeddings = self.embed_layer(input_wave)
        # embeddings is [N,F,T]
        enc_output = self.encoder_layer(embeddings)
        # enc_output is [N,F,T]
        mask = self.mask_layer(enc_output)
        dec_output = self.decoder_layer(mask, enc_output)
        # mask is [N,F,T]        
        
        output = self.output_layer(dec_output)
        if False:
            print_meta("input_wave", input_wave)
            print_meta("embeddings", embeddings)
            print_meta("enc_output", enc_output)
            print_meta("mask", mask)
            print_meta("dec_output", dec_output)
            print_meta("output", output)
        return output

    def loss_function(self,
                      y:Tensor,
                      yhat:Tensor,
                      *args,
                      **kw) -> dict:
        results = []
        x = kw['x']
        #wndyhat = self.loss_window * yhat
        #wndy = self.loss_window * y
        #wnd_sisnr = torch.mean(evaluate.sisnr(wndy, wndyhat))

        sisnr = torch.mean(metrics.sisnr(y, yhat))
        sisnrX = torch.mean(metrics.sisnr(x, yhat))
        sisnri = torch.mean(sisnr - metrics.sisnr(y, x))
        sdr = torch.mean(metrics.new_sdr(y, yhat))
        mse = F.mse_loss(yhat, y)
        mean = torch.mean(yhat)-torch.mean(y)
        std = torch.std(yhat)-torch.std(y)
        
        results.append(("loss",-sisnr))

        results.append(("sisnr",sisnr))
        results.append(("sisnri",sisnri))
        results.append(("sdr",sdr))
        results.append(("mse",mse))
        results.append(("mean",mean))
        results.append(("std",std))
        results.append(("sisnrX",sisnrX))

        return results

if __name__ == '__main__':
    model = Model()
    from torchinfo import summary
    summary(model, (16,8000))
    print(model.__dict__)

