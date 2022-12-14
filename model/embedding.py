import torch
from torch import nn, Tensor
import torch.nn.functional as F


""" 
nn.Module that convert an audio segment to a list of embeddings with specific window_size and hop_size 
"""

class EmbeddingSTFT(nn.Module):
    """
        Use stft to extract embeddings of wave features
    """
    def __init__(self, 
                    wave_length=8000,   # L
                    window_size=512,    # n_fft
                    hop_size=160,       # H
                    num_features=512,
                    ):
        """
            input shape:  (N,L)
            output shape: (N,C,Lout)
        """
        super(EmbeddingSTFT, self).__init__()
        self.wave_length = wave_length
        self.window_size = window_size
        self.hop_size = hop_size
        self.padding = (hop_size)//2
        self.num_features = num_features

        self.n_fft = self.window_size

        self.F = num_features
        self.T = wave_length // hop_size

        self.conv = nn.Conv1d(
            in_channels=self.window_size,
            out_channels=self.num_features,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        #self.layernorm = nn.LayerNorm((self.num_features, self.T))
        self.layernorm = nn.LayerNorm((self.T))
        self.prelu = nn.PReLU(self.num_features)

        self.embedding_dim = self.num_features
        self.raw = None

    def forward(self, input:Tensor):
        x = input
        assert x.dim() == 2 and x.shape[-1] == self.wave_length, f"x is {x.shape}, but require {('N',self.wave_length)}"
        x = F.pad(x, (self.padding,self.padding))
        z = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft).to(x),
            win_length=self.n_fft,
            normalized=True,
            center=True,
            return_complex=False,
            pad_mode='reflect',
        )
        # remove the 0th freq bin, remove the 0th and last time
        z = z[:,1:,1:-1,:]
        # z is [N,F//2,T,2] here (because not returning complex)
        z = z.permute(0,1,3,2)
        # z is changed to [N,F//2,2,T] here
        z = torch.flatten(z, 1, 2)
        # z is changed to [N,F,T]

        self.raw = z.clone()

        z = self.conv(z)
        z = self.layernorm(z)
        z = self.prelu(z)
        return z

class DecodeSTFT(nn.Module):
    """
        Decode STFT embeddings to wave
    """
    def __init__(self, embed:EmbeddingSTFT):
        super(DecodeSTFT, self).__init__()
        self.embed = embed
        self.T = embed.T
        self.F = embed.F
        self.n_fft = embed.n_fft
        self.hop_size = embed.hop_size
        self.padding = embed.padding
        self.num_features = embed.num_features

        self.convtrans = nn.ConvTranspose1d(
            in_channels=self.num_features,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, z:Tensor):
        z = self.convtrans(z) 

        z = z.view(z.shape[0], z.shape[1]//2, 2, z.shape[2])
        z = z.permute(0,1,3,2) #[N,F,T,2]
        z = F.pad(z, (0,0,1,1,1,0))
        x = torch.istft(
            z,
            self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft).to(z.real),
            normalized=True,
            center=True,
            return_complex=False,
        )
        x = x[:,self.padding:-self.padding]
        return x

class EmbeddingConv(nn.Module):
    """
        Use Conv1d to make embedded audio segments
    """
    def __init__(self, 
                    wave_length=8000,   # L
                    window_size=512,    # W 
                    hop_size=160,       # H
                    num_features=512, # C out channels in conv
                    ):
        """
            input shape:  (N,L) or (N,1,L)
            output shape: (N,C,Lout)
        """
        super(EmbeddingConv, self).__init__()
        self.wave_length = wave_length
        self.window_size = window_size
        self.hop_size = hop_size
        self.num_features = num_features

        self.targetLout = wave_length // hop_size
        self.padding = ((self.targetLout-1) * self.hop_size + window_size - wave_length + 1) // 2
        self.Lout = int((wave_length + 2 * self.padding - (self.window_size-1) -1) / self.hop_size + 1)
        assert self.Lout == self.targetLout
        self.embedding_dim = self.num_features
        self.T = wave_length // hop_size

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.num_features,
            kernel_size=self.window_size,
            stride=self.hop_size,
            padding=self.padding,
        )
        #self.layernorm = nn.LayerNorm((self.num_features, self.T))
        self.layernorm = nn.LayerNorm((self.T))
        self.prelu = nn.PReLU(self.num_features)

        self.raw = None

    def forward(self, input:Tensor):
        x = input
        assert x.shape[-1] == self.wave_length
        if x.dim() == 2: # [B,L]
            x = x.view(x.shape[0], 1, x.shape[1])
        x = self.conv(x)
        output = self.layernorm(x)
        output = self.prelu(x)
        self.raw = output.clone()
        return output

class DecodeConv(nn.Module):
    """
        Decode Conv embeddings to wave
    """
    def __init__(self, embed:EmbeddingConv):
        super(DecodeConv, self).__init__()

        self.convtranspose = nn.ConvTranspose1d(
            in_channels=embed.num_features,
            out_channels=1,
            kernel_size=embed.window_size,
            stride=embed.hop_size,
            padding=embed.padding,
        )
        #self.activation = nn.Sigmoid()

    def forward(self, z:Tensor):
        z = self.convtranspose(z)
        #z = self.activation(z)
        # z is [N,1,L]
        z = z.view(z.shape[0], z.shape[2])
        return z
    
class EmbeddingHybrid(nn.Module):
    """
        Use hybrid method to extract embeddings of wave features
    """
    def __init__(self, 
                    wave_length=8000,   # L
                    window_size=512,    # n_fft
                    hop_size=160,       # H
                    num_features=512,
                    ):
        """
            input shape:  (N,L)
            output shape: (N,C,Lout)
        """
        super(EmbeddingHybrid, self).__init__()
        self.wave_length = wave_length
        self.window_size = window_size
        self.hop_size = hop_size
        self.stft_padding = (hop_size)//2
        self.num_features = num_features

        self.n_fft = self.window_size

        self.F = self.num_features
        self.T = wave_length // hop_size

        self.targetLout = wave_length // hop_size
        self.wave_padding = ((self.targetLout-1) * self.hop_size + window_size - wave_length + 1) // 2
        self.Lout = int((wave_length + 2 * self.wave_padding - (self.window_size-1) -1) / self.hop_size + 1)
        assert self.Lout == self.targetLout

        self.conv1 = nn.Conv1d(
            in_channels=self.window_size,
            out_channels=self.num_features//2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=self.num_features//2,
            kernel_size=self.window_size,
            stride=self.hop_size,
            padding=self.wave_padding,
        )
        self.layernorm = nn.LayerNorm((self.T))
        self.prelu = nn.PReLU(self.num_features)

        self.embedding_dim = self.num_features
        self.raw = None
    def forward(self, input:Tensor):
        x1 = input
        x1 = F.pad(x1, (self.stft_padding,self.stft_padding))
        z = torch.stft(
            x1,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft).to(x1),
            win_length=self.n_fft,
            normalized=True,
            center=True,
            return_complex=False,
            pad_mode='reflect',
        )
        # remove the 0th freq bin, remove the 0th and last time
        z = z[:,1:,1:-1,:]
        # z is [N,F//2,T,2] here (because not returning complex)
        z = z.permute(0,1,3,2)
        # z is changed to [N,F//2,2,T] here
        z = torch.flatten(z, 1, 2)
        # z is changed to [N,F,T]
        z = self.conv1(z)

        x2 = input
        assert x2.shape[-1] == self.wave_length
        if x2.dim() == 2: # [B,L]
            x2 = x2.view(x2.shape[0], 1, x2.shape[1])
        x2 = self.conv2(x2) 

        output = torch.cat((z, x2), 1)
        return output
        
class DecodeHybrid(nn.Module):
    """
        Decode Hybrid embeddings to wave
    """
    def __init__(self, embed:EmbeddingHybrid):
        super(DecodeHybrid, self).__init__()
        self.embed = embed
        self.T = embed.T
        self.F = embed.F
        self.n_fft = embed.n_fft
        self.hop_size = embed.hop_size
        self.stft_padding = embed.stft_padding
        self.wave_padding = embed.wave_padding
        self.num_features = embed.num_features

        self.convtrans1 = nn.ConvTranspose1d(
            in_channels=self.num_features//2,
            out_channels=self.n_fft,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.convtrans2 = nn.ConvTranspose1d(
            in_channels=embed.num_features//2,
            out_channels=1,
            kernel_size=embed.window_size,
            stride=embed.hop_size,
            padding=embed.wave_padding,
        )

    def forward(self, z:Tensor):
        f = self.num_features // 2
        z1 = z[:,:f,:]
        z2 = z[:,f:,:]

        z1 = self.convtrans1(z1) 
        z1 = z1.view(z1.shape[0], z1.shape[1]//2, 2, z1.shape[2])
        z1 = z1.permute(0,1,3,2) #[N,F,T,2]
        z1 = F.pad(z1, (0,0,1,1,1,0))
        x1 = torch.istft(
            z1,
            self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft).to(z1.real),
            normalized=True,
            center=True,
            return_complex=False,
        )
        x1 = x1[:,self.stft_padding:-self.stft_padding]

        x2 = self.convtrans2(z2)
        x2 = x2.view(x2.shape[0], x2.shape[2])
        return (x1+x2)/2

if __name__ == '__main__':
    from torchinfo import summary

    if False:
        ms = EmbeddingSTFT()
        ds = DecodeSTFT(ms)
        fulls = nn.Sequential(ms,ds)
        summary(fulls,(8,ms.wave_length))

        mc = EmbeddingConv()
        dc = DecodeConv(mc)
        fullc = nn.Sequential(mc,dc)
        summary(fullc, (8,mc.wave_length))

    mh = EmbeddingHybrid()
    dh = DecodeHybrid(mh)
    fullh = nn.Sequential(mh,dh)
    summary(fullh, (8,mh.wave_length))