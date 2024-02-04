# %%
import os
import json
import gzip

from tqdm import tqdm
import numpy as np
import pandas as pd

from tools import save_gzip_dict, txtfile_to_graphdict
# %%
def prepare_secondary_df(file) -> dict:
    datass = pd.read_pickle(file, compression='gzip')
    datass.set_index('sequence', inplace=True)
    datass = datass.to_dict('index')
    return datass

train_mode = True
rnadir = "/home/nfs/kkaminski/kaggle/"
# %%
dfile = os.path.join(rnadir, "train_data.p.gz")
dfile_test = os.path.join(rnadir, "test_sequences.p.gz")
dfile_sec =  os.path.join(rnadir, "train_secondary.p.gz")
dfile_sec_test = os.path.join(rnadir, "test_secondary.p.gz")

basedir = "/home/nfs/kkaminski/kaggle/Ribonanza_bpp_files/extra_data"
outdir = "/home/nfs/kkaminski/kaggle/structure"

outdir_train = os.path.join(outdir, 'train')
outdir_test = os.path.join(outdir, 'test')

os.makedirs(outdir_test, exist_ok=True)
os.makedirs(outdir_train, exist_ok=True)
# %%
print('preparing secondary')
if train_mode:
    datass = prepare_secondary_df(dfile_sec)
else:
    datass = prepare_secondary_df(dfile_sec_test)
# %%
if train_mode:
    dataseq = pd.read_pickle(dfile, compression='gzip')
    dataseq = dataseq.loc[dataseq.SN_filter == 1]
    dataseq.drop_duplicates(['sequence'], inplace=True)
    dataseq['sequence_id'] = dataseq['sequence_id'].str.lower()
    dataseq.set_index('sequence_id', inplace=True)
    dataseq = dataseq.to_dict('index')
    print('train sequences loaded:', len(dataseq))
else:
    dataseq = pd.read_pickle(dfile_test, compression='gzip')
    dataseq['sequence_id'] = dataseq['sequence_id'].str.lower()
    dataseq.set_index('sequence_id', inplace=True)
    dataseq = dataseq.to_dict('index')
    print('test sequences loaded:', len(dataseq))
# %%
assert os.path.isdir(basedir)
chunk_nb = 0
chunksize = 12800
skipped = 0
total = 0
total_train = 0
total_test = 0
fname = "train" if train_mode else "test"
dictlog = list()
dictstruct = dict()
print('mode: ', fname)
with tqdm(total=len(dataseq)) as pbar:
    for itr, (dirpath, dir, files) in enumerate(os.walk(basedir)):
        if len(files) == 0:
            continue
        absfiles = [os.path.join(dirpath, f) for f in files]
        for f in absfiles:
            filename = os.path.basename(f)
            seqid, _ = os.path.splitext(filename)
            seqid = seqid.lower()
            
            record = {'sequence_id' : seqid,
                    'source': filename,
                    }
            is_valid = True
            is_train = True
            if seqid in dataseq:
                sequence = dataseq[seqid]['sequence']
                secondary = datass[sequence]['secondary']
                total += 1
                pbar.update(1)
            else:
                is_valid = False
                skipped += 1
            if not is_valid:
                continue
            record_struct = txtfile_to_graphdict(f)
            record['is_valid'] = is_valid
            record['is_train'] = is_train
            dictlog.append(record)
            record_struct['sequence'] = sequence
            record_struct['secondary'] = secondary
            dictstruct[seqid] = record_struct
            if len(dictstruct) == chunksize:
                filename = os.path.join(outdir_train, f'{fname}.jzon.gz.{chunk_nb}')
                save_gzip_dict(dictstruct, filename)
                chunk_nb += 1
                del dictstruct
                dictstruct = dict()
        pbar.set_description(f'skipped: {skipped}/{total}')
        
# save last
filename = os.path.join(outdir_train, f'{fname}.jzon.gz.{chunk_nb}')
save_gzip_dict(dictstruct, filename)

dflog = pd.DataFrame(dictlog)
dflog.to_csv(os.path.join(outdir, f'parselog_{fname}.csv'))
print('train structures saved', len(dictstruct))
print('Done')
