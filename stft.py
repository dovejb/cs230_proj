from torch import nn
import torch
from typing import Tuple
class STFT(nn.Module):
    def __init__(self,
                 n_fft: int = 512) -> None:
        super(STFT, self).__init__()
        self.n_fft = n_fft

    def forward(self, x:torch.Tensor):
        """
        :param x: [N, 1, L]
        :return: [N, 2, F, T]
        """
        assert x.dim() == 3
        n, _, l = x.shape
        x = x.view(n, l)
        z = torch.stft(x, 
                       self.n_fft,
                       hop_length=self.n_fft // 4,
                       window=torch.hann_window(self.n_fft).to(x),
                       win_length=self.n_fft,
                       normalized=True,
                       center=True,
                       return_complex=False,
                       pad_mode='reflect')
        # here z is [N, F, T, 2]
        _, freqs, frames, channels = z.shape
        # need [N, 2, F, T]
        z = z.permute(0, 3, 1, 2)
        # make F&T even
        return z[:,:,1:,:-1]

class ISTFT(nn.Module):
    def __init__(self):
        super(ISTFT, self).__init__()

    def forward(self, z:torch.Tensor):
        """
        :param Z: [N, 2, F, T]
        :return: [N, 1, L]
        """
        assert z.dim() == 4
        # add the axis dropped when STFT
        z = torch.nn.functional.pad(z, (0,1,1,0))
        # need z [N,F,T,2]
        z = z.permute(0,2,3,1)
        n, freqs, _, _ = z.shape
        n_fft = 2 * freqs - 2
        win_length = n_fft
        x = torch.istft(z,
                        n_fft,
                        hop_length=None,
                        window=torch.hann_window(win_length).to(z.real),
                        normalized=True,
                        length=None,
                        center=True,
                        return_complex=False,
                        )
        _, length = x.shape
        return x.view(n, 1, length)


if __name__ == '__main__':
    from datasets import *
    x, y = load_test_datasets()
    x = torch.from_numpy(x[...,:352768])
    print(x.shape)
    stftm = STFT()
    tz = stftm.forward(x)
    print(tz.shape)
    istftm = ISTFT()
    x_ = istftm.forward(tz)
    print(x_.shape)
    loss = nn.MSELoss()
    print("loss1", loss(x_, x))
    print("loss2", loss(x, x_))
    print("tz mean", torch.mean(tz))
    print("x mean", torch.mean(x))
    print("y mean", np.mean(y))