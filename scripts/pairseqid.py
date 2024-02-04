import pickle
import gzip
import json
import os

from tqdm import tqdm
import pandas as pd

from tools import load_gzip_dict, save_gzip_dict


rna_filt_data = "/home/nfs/kkaminski/kaggle/rna_filt80"
parsedata = "/home/nfs/kkaminski/kaggle/rna_filt80/{exp_type}/sequence_id.csv"
data = "/home/nfs/kkaminski/kaggle/train_data.p"
datadms = "/home/nfs/kkaminski/kaggle/rna_filt/dms/sequence_id.csv"
graphdir = "/home/nfs/kkaminski/kaggle/structure/train.json.gz."
grapfilepref = "/home/nfs/kkaminski/kaggle/structure/train/train.json.gz."
graphout = "/home/nfs/kkaminski/kkagle/graphtrain/"
graphlog = "/home/nfs/kkaminski/kaggle/structure/parselog_train.csv"
#data = pd.read_pickle(data, compression='gzip')

glog = pd.read_csv(graphlog)


glog['index'] = list(range(0, glog.shape[0]))
glog['fileid'] = (glog['index'] // 12800).astype(str)
glog['file'] = glog['fileid'].apply(lambda x: grapfilepref + x)
#graphlog.set_index('sequence_id', inplace=True)


output_dms = os.path.join(rna_filt_data, 'dms', 'structdata.json.gz')
output_2a3 = os.path.join(rna_filt_data, '2a3', 'structdata.json.gz')
files = glog['file'].unique().tolist()
seqid_dms = pd.read_csv(parsedata.format(exp_type='dms'))
seqid_2a3 = pd.read_csv(parsedata.format(exp_type='2a3'))
seqid_dms = {seqid: None for seqid in seqid_dms['sequence_id'].tolist()}
seqid_2a3 = {seqid: None for seqid in seqid_2a3['sequence_id'].tolist()}
for f in files:
    print(f)
    gdata = load_gzip_dict(f)
    for seqid, gvals in gdata.items():
        if seqid in seqid_dms:
            seqid_dms[seqid] = gvals
        if seqid in seqid_2a3:
            seqid_2a3[seqid] = gvals

for seqid, gvals in seqid_2a3:
    assert gvals is not None

for seqid, gvals in seqid_dms:
    assert gvals is not None

save_gzip_dict(seqid_dms, output_dms)
save_gzip_dict(seqid_2a3, output_2a3)


