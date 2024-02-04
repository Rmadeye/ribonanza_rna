import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RNA_Model(nn.Module):
    _num_labels: int = 5
    def __init__(self, dim=128, depth=4, heads=None, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(self._num_labels, dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        heads = dim//head_size
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Linear(dim,1)
    
    def forward(self, seq, mask):

        Lmax = mask.shape[1]
        pos = torch.arange(Lmax, device=seq.device).unsqueeze(0)
        # [batch, seqlen, dim]
        pos = self.pos_enc(pos)
        # [batch, seqlen, dim]
        x = self.emb(seq)
        x = x + pos

        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)
        return x.squeeze(-1)
