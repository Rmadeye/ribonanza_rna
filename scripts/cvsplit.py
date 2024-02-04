import os
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold


for exptype in ['2a3', 'dms']:
    clustdata = f'/home/nfs/rmadaj/ai/kaggle/kaggle_rna/data/filter/80/{exptype}/clustered_{exptype}.csv'
    outdir = f"/home/nfs/rmadaj/ai/kaggle/kaggle_rna/data/filter/80/{exptype}"
    fname = os.path.join(outdir, 'fold_ids.json')
    clustdf = pd.read_csv(clustdata)
    clusters = clustdf.cluster.values
    ids = np.arange(0, clustdf.shape[0])

    group_kfold = GroupKFold(n_splits=4)
    folds = dict()
    for i, (train_index, test_index) in enumerate(group_kfold.split(ids, groups=clusters)):
        folds[i] = {
            'ids' : train_index.tolist(),
            'ids_test' : test_index.tolist()
            }
        
        
    with open(fname, 'wt') as fp:
        json.dump(folds, fp)
    print('saved: ', fname)
