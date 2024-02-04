import os
import argparse

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('preds', type=str)
parser.add_argument('-nfolds', default=4, type=int)

args = parser.parse_args()

def fold_iter(path, fold, chunksize = 128000):

    path = f"{path}/{fold}.csv"
    assert os.path.isfile(path)
    return pd.read_csv(path, chunksize=chunksize)


folds = zip(
    fold_iter(args.preds, 1),
      fold_iter(args.preds, 2),
        fold_iter(args.preds, 3),
        fold_iter(args.preds, 4))

mergefile = os.path.join(args.preds, 'merged.csv.gz')
ids = list()
reacts = list()
react_cols = ['reactivity_DMS_MaP','reactivity_2A3_MaP']
for itr, (f1, f2, f3, f4) in enumerate(folds):
    
    ids.extend(f1['id'].tolist())
    vals = f1[react_cols].values + f2[react_cols].values + f3[react_cols].values + f4[react_cols].values
    vals /= 4
    reacts.append(vals)
    resdf = pd.DataFrame()
    resdf['id'] = f1['id'].tolist()
    resdf[react_cols[0]] = vals[0, :]
    resdf[react_cols[1]] = vals[1, :]
    mode = 'w' if itr != 0 else 'a'
    header = itr == 0
    resdf.to_csv(mergefile, mode=mode, header=header, compression='gzip')

    