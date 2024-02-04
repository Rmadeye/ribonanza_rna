import sys
import os

import dgl
import torch
from torch import nn
from torch.nn import functional as F

PATH_MODULE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH_MODULE)

#from NetworkUtils.GraphNetworks.edge_gat_layer import EGATConv
from dgl.nn import EGATConv


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M


class LayerBlock(nn.Module):
    def __init__(self, in_dim_n, in_dim_e, heads,
                 residual=False, **kw_args):
        super().__init__()    
        self.num_heads = heads
        out_dim_n = int(in_dim_n/heads)
        out_dim_e = int(in_dim_e/heads)
        assert out_dim_n != 0, f' {in_dim_n}/{heads} == {out_dim_n}'
        assert out_dim_e != 0, f' {in_dim_e}/{heads} == {out_dim_e}'
        self.bn_node = nn.BatchNorm1d(in_dim_n)
        self.bn_edge = nn.BatchNorm1d(in_dim_e)
        self.egatconv = EGATConv(in_node_feats=in_dim_n,
                                  in_edge_feats=in_dim_e, 
                                  out_node_feats=out_dim_n,
                                  out_edge_feats=out_dim_e,
                                  num_heads=heads)

    def forward(self, g, h, e):
        h, e = self.egatconv(g, h, e)
        h, e = h.flatten(1),  e.flatten(1)
        h, e = self.bn_node(h), self.bn_edge(e)
        h = F.leaky_relu(h) 
        e = F.leaky_relu(e)
        return h, e

class RNAGAT(nn.Module):
    seqdict_size = 21
    dsspdict_size = 3
    n_classes = 4
    dimedge = 1
    def __init__(self,
                 seqdim, ssdim, residual = True, depth=8,
                  num_heads = 4, hidden_dim = 128, hidden_dim_edge = 32):
        super(RNAGAT, self).__init__()
        
        assert seqdim > -1
        assert ssdim > -1
        

        self.indim = seqdim + ssdim
        self.hidden_dim_edge = hidden_dim_edge
        self.depth = depth
        self.num_heads = num_heads
        self.residual = bool(residual)
        # hidden sizes
        self.in_dim =  seqdim + ssdim

        #### embedding layers
        self.seqemb = nn.Embedding(self.seqdict_size, seqdim)
        self.secemb = nn.Embedding(self.dsspdict_size, ssdim)
        self.egatconv = EGATConv(in_node_feats=self.indim,
                                  in_edge_feats=1,
                                  out_node_feats=self.indim,
                                  out_edge_feats=self.hidden_dim_edge,
                                  num_heads=1,
                                  bias=True)
        self.GATBlocks = nn.ModuleList()
        for _ in range(self.depth):
            self.GATBlocks.append(
                LayerBlock(in_dim_n=self.in_dim, in_dim_e=self.hidden_dim_edge,
                heads=self.num_heads)
            )      
        self.fc = nn.Linear(self.in_dim, 1)

    def encoder(self, g):
        '''
        extract embeddings
        Returns:
            node_embedding (torch.FloatTensor)
        '''
        #### extract feature vector concat embeddings with handcrafted features
        emb_seq = self.seqemb(g.ndata['sequence'])
        emb_sec = self.secemb(g.ndata['secondary'])
        e = g.edata['p'].view(-1, 1)
        h = torch.cat([emb_seq, emb_sec], dim=1)
        h, e = self.egatconv(g, h, e)
        ###### loop layer block #############
        for block in self.GATBlocks:
            h, e = block(g, h, e)
        return h

    def unbatch(self, g, feats):
        """
        convert features to list of tensors
        """
        gsizes = g.batch_num_nodes().tolist()
        return torch.split(feats, gsizes, 0)

    def forward(self, g):
        feats = self.encoder(g)
        reactivity = self.fc(feats)
        return reactivity
