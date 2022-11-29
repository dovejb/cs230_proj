from torch import nn, Tensor
import torch
from utils import pos_encoding, get_attn_local_mask
from embedding import *

""" Multi-Head Attention encoder"""

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, fc_dim, num_heads, dropout, T, mask):
        super(EncoderLayer, self).__init__()
        self.mask = mask
        self.mha = nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True)
        # The dimension is changed to [N,T,F] here
        self.layernorm1 = nn.LayerNorm([T, embed_dim])
        self.layernorm2 = nn.LayerNorm([T, embed_dim])
        self.ffn = nn.Sequential(
                        nn.Linear(embed_dim,fc_dim),
                        nn.Linear(fc_dim,embed_dim),
                    )
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, input:Tensor):
        x = input
        attn_output = self.mha(query=x,value=x,key=x,attn_mask=self.mask)[0]

        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_ffn(ffn_output)

        output = self.layernorm2(out1 + ffn_output)
        return output

class Encoder(nn.Module):
    def __init__(self,
                embedding_layer:nn.Module,
                num_heads=8,
                num_layers=2,
                dropout=0.0,
                local_size=5,
                ffn_dim=None,
                ):
        super(Encoder, self).__init__()
        self.T = embedding_layer.T
        self.num_heads = num_heads
        self.dropout = dropout
        self.embed_dim = embedding_layer.embedding_dim
        self.sqrt_embdim = torch.sqrt(torch.Tensor([float(self.embed_dim)])).cuda()
        self.num_layers = num_layers
        self.ffn_dim = self.embed_dim if ffn_dim is None else ffn_dim
        self.attn_mask = get_attn_local_mask(local_size,self.T)

        self.embedding_layer = embedding_layer
        self.encoderLayers = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.encoderLayers.append(EncoderLayer(self.embed_dim, 
                                                    self.ffn_dim,
                                                    self.num_heads,
                                                    self.dropout,
                                                    self.T,
                                                    self.attn_mask))

    def forward(self,input:Tensor):
        """
            input: Tensor in shape (N,L)
        """
        assert input.shape[-1] == self.embedding_layer.wave_length

        x = self.embedding_layer(input)

        N,F,T = x.shape
        # scale x as the course does
        x *= self.sqrt_embdim
        # add position encoding
        x += pos_encoding[:N,:F,:T]
        x = x.permute(0, 2, 1)
        x = x.to(torch.float)
        for i in range(self.num_layers):
            x = self.encoderLayers[i].forward(x)
        return x


if __name__ == '__main__':
    embedding = EmbeddingSTFT()
    encoder = Encoder(
        embedding_layer=embedding,
    )
    from torchinfo import summary
    summary(encoder, (16,8000))