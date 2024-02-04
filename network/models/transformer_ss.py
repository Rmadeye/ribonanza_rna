import torch
import torch.nn as nn
import math
# import torchviz

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

class SecStructModel(nn.Module):
    _num_sslabels: int = 4
    _num_seqlabels: int = 5
    def __init__(self, seq_dim=128, depth=4, heads=6, dropout_transformer=0.1,**kwargs):
        super().__init__()
        self.seq_dim = seq_dim
        self.dropout_transformer = dropout_transformer
        self.ss_dim = seq_dim // 2  # New dimension for secondary structure
        self.indim = self.seq_dim + self.ss_dim
        # now make pos_enc match  seq_dim and ss_dim concatenated
        self.pos_enc = SinusoidalPosEmb(self.indim) 
        self.emb = nn.Embedding(self._num_seqlabels, seq_dim)
        self.emb_ss = nn.Embedding(self._num_sslabels, self.ss_dim)  # Embedding for secondary structure
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.indim, nhead=heads, dim_feedforward=4*self.indim,
                dropout=self.dropout_transformer, activation=nn.GELU(), batch_first=True, norm_first=True), depth, norm=nn.LayerNorm(self.indim))
        self.dropout = nn.Dropout(0.25)
        self.proj_out = nn.Linear(self.indim,1)
        
    
    def forward(self, seq, ss, mask):

        pos = torch.arange(mask.shape[1], device=seq.device).unsqueeze(0)
        # [batch, seqlen, dim]
        pos = self.pos_enc(pos)
        # [batch, seqlen, dim]
        emb_seq = self.emb(seq)
        emb_ss = self.emb_ss(ss)
        embeddings = torch.cat((emb_seq, emb_ss),dim=-1)
        embeddings = embeddings + pos
        x = self.transformer(embeddings, src_key_padding_mask=~mask)
        x = self.dropout(x)
        x = self.proj_out(x)
        
        return x.squeeze(-1)
