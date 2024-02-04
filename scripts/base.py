import os

import torch

from tools import load_gzip_dict
structdir = '/home/nfs/kkaminski/structrue/train'

structfile = os.path.join(structdir, 'train.json.gz.{chunk_nb}')

num_chunks = 10

dict_struct = dict()
for i in range(num_chunks):
    chunk_data = load_gzip_dict(structfile.format(chunk_nb=i))
    dict_struct.update(chunk_data)