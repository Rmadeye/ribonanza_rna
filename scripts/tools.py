import os
import pandas as pd
import json
import gzip

import torch
import dgl

rna_dict = dict(A=0, C=1, G=2, U=3)
rna_map = lambda letter : rna_dict[letter]
ss_dict = {'(': 0, ')' : 1, '.': 2}
ss_map = lambda letter : ss_dict[letter]


def encode_seq_data(sequence, secondary):
  return torch.LongTensor(list(map(rna_map, sequence))), torch.LongTensor(list(map(ss_map, secondary)))


def save_gzip_dict(listdict, filename):
    # https://stackoverflow.com/questions/39450065/python-3-read-write-compressed-json-objects-from-to-gzip-file
    json_obj = json.dumps(listdict)
    json_obj += '\n'
    with gzip.GzipFile(filename, mode='wb') as fp:
        fp.write(json_obj.encode('utf-8'))


def load_gzip_dict(filename):
    assert os.path.isfile(filename)
    with gzip.GzipFile(filename, mode='rb') as fp:
        json_obj = fp.read()
    json_obj = json.loads(json_obj)
    return json_obj


def txtfile_to_graphdict(file):
    data = pd.read_csv(file, sep=" ", header=None, names=['u','v','p'])
    u, v, p = data['u'].values, data['v'].values, data['p'].values
    u, v, p = u.tolist(), v.tolist(), p.round(3).tolist()
    _struct = {
        'u' : u, 
        'v':  v,
        'p': p
    }
    return _struct


def seqlist_to_fasta_str(ids, seqlist) -> str:
    """
    return file name
    """
    string = ""
    for idx, seq in zip(ids, seqlist):
        string += f"> {idx}\n"
        string += f"{seq}\n"
    return string

def graphdict_to_dgl(graphdict: dict) -> dgl.graph:
    '''
    requires u,v,p, sequence, secondary keys
    '''

    sequence, secondary = graphdict['sequence'], graphdict['secondary']
    u, v, p = graphdict['u'], graphdict['v'], graphdict['p']
    u, v, p = torch.LongTensor(u), torch.LongTensor(v), torch.LongTensor(p)
    assert len(sequence) == len(secondary)
    sequence, secondary = encode_seq_data(sequence, secondary)
    seqlen = sequence.shape[0]
    contact_matrix = torch.zeros((seqlen, seqlen))
    ubase, vbase = torch.arange(0, seqlen-1), torch.arange(1, seqlen)
    contact_matrix[ubase, vbase] = 1
    contact_matrix[vbase, ubase] = 1
    # u, v are numbered from 1
    u -= 1
    v -= 1
    contact_matrix[u, v] = 1
    contact_matrix[v, u] = 1

    ugprah, vgraph = torch.where(contact_matrix == 1)
    connection_type = (ugprah - vgraph) == 1
    graph = dgl.graph((ugprah, vgraph))
    graph.ndata['sequence'] = sequence
    graph.ndata['secondary'] = secondary
    graph.edata['p'] = connection_type*1.0

    return graph

