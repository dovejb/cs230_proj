from torch import nn, Tensor
import torch
from .embedding import *
from .utils import get_attn_local_mask

""" Multi-Head Attention decoder"""

class Decoder(nn.Module):
    def __init__(self, embed_dim, T, num_heads, dropout, local_size):
        super(Decoder, self).__init__()
        self.T = T
        self.attn_mask = get_attn_local_mask(local_size,self.T)
        self.mha = nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        )
        self.relu = nn.ReLU()
        self.convtrans = nn.ConvTranspose1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, mask:Tensor, encodings:Tensor):
        """
            mask: the output of mask_layer, shape [N, F, T]
            encodings: the output of encoder_layer(reshaped), shape should be [N, F, T]
        """
        mask = mask.permute(0,2,1)
        attn_output = self.mha(query=mask,value=mask,key=mask,attn_mask=self.attn_mask)[0]
        attn_output = attn_output.permute(0,2,1)
        attn_output = self.relu(attn_output)

        output = attn_output * encodings

        output = self.convtrans(output)

        return output

if __name__ == '__main__':
    from torchinfo import summary