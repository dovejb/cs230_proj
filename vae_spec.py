import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from stft import STFT, ISTFT
from torchinfo import summary
import numpy as np

from typing import List, Tuple

class SpecVAE(nn.Module):
    def __init__(self,
                 wav_length: int,
                 latent_dim: int,
                 beta: float = 0.01,
                 hidden_dims: List = None) -> None:
        super(SpecVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        self.num_iter = 0

        modules = []
        if hidden_dims is None:
            hidden_dims = [4,16,32,64,128]
        self.hidden_dims = hidden_dims

        modules.append(
            STFT(),
        )

        in_channels = 2 # don't return complex, and it's always 2 after STFT
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 
                              out_channels=h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        #summary(self.encoder, (32,1,352256))
        # use summary to get this magic number!
        self.shape_before_bottleneck = [128, 8, 86] # [C,H,W]
        flatten_size = np.prod(self.shape_before_bottleneck)
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_var = nn.Linear(flatten_size, latent_dim)


        # for decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, flatten_size)
        modules.append(
            Reshape(self.shape_before_bottleneck)
        )
        hidden_dims.reverse()

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i+1],
                                        kernel_size=3, 
                                        stride=2,
                                        padding=1,
                                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                                hidden_dims[-1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], 
                        out_channels=2,
                        kernel_size=3,
                        padding=1,
            ),
            ISTFT(),
        )
    
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        :param input: (Tensor) [N, 1, L]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of latent Gaussian
        :param logvar: (Tensor) Stanrdard deviation of latent Gaussian
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      y,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        recons_loss = F.mse_loss(recons, y)
        kld_loss = torch.mean(-0.5 * torch.sum(1+log_var-mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.beta * kld_loss

        return {'loss': loss, 'Reconstruction-Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x:Tensor):
        size = np.prod(x.shape)
        N = size // np.prod(self.shape)
        return x.view(N, *self.shape)