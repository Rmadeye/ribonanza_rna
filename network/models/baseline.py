import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class RNABaseline(nn.Module):
    DICT_SIZE: int = 5
    OUTPUT_DIM: int = 1
    def __init__(self, input_dim: int = 64, hidden_dim: int =64, *args, **kw_args):
        super().__init__()
        # definition of nn layers
        self.embedding = nn.Embedding(num_embeddings=self.DICT_SIZE,
                                       embedding_dim=input_dim)
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=6, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.OUTPUT_DIM)
        
    def forward(self, seqs):
        # input [batch_size, max_seqlen]
        # emb  [batch_size, max_seqlen, input_dim]
        x = self.embedding(seqs)
        # lstm input (emb, prev_hidden_dim)
        # prev_hidden_state (batch_size, max_seqlen, 1)
        out, hidden_state = self.lstm(x)
        # lstm output (emb, max_seqlen, hidden_dim)
        # linear output (emb, max_seqlen, 1)
        # after squeeze (emb, max_seqlen)
        out = self.linear(out).squeeze(-1)
        return out
