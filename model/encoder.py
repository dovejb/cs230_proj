from torch import nn, Tensor
import torch
from .utils import pos_encoding, get_attn_local_mask
from .embedding import *

""" Multi-Head Attention encoder"""

class Encoder(nn.Module):
    def __init__(self, embed_dim, T, num_heads, dropout, local_size):
        super(Encoder, self).__init__()
        self.T = T
        self.attn_mask = get_attn_local_mask(local_size,self.T)
        self.mha = nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, input:Tensor):
        x = input # input is [N,F,T]
        x = x.permute(0,2,1)

        attn_output = self.mha(query=x,value=x,key=x,attn_mask=self.attn_mask)[0]
        attn_output = attn_output.permute(0,2,1) # to [N,F,T]

        output = attn_output * input
        output = self.relu(output)

        return output

if __name__ == '__main__':
    embedding = EmbeddingSTFT()
    encoder = Encoder(
        embedding_layer=embedding,
    )
    from torchinfo import summary
    summary(encoder, (16,8000))