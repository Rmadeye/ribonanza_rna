import os
import time
from typing import List
import argparse
from tqdm import tqdm
import shutil
import json

import pandas as pd
import numpy as np
import torch

from data_version import PATH_SS_TRAIN, PATH_SS_TEST, PATH_SEQ_TEST

rna_dict = dict(A=0, C=1, G=2, U=3)
rna_map = lambda letter : rna_dict[letter]
ss_dict = {'(', ')', '-'}
ss_map = lambda letter : ss_dict[letter]


def main(args: argparse.Namespace):

    # params
    # clip reactivity to range (-q99, q99)
    # q99 = 1.1
    # q99 should be between 0 and 1
    qmin = 0.0
    qmax = 1.0

    PATH_BOTH = 'both'
    PATH_TEST = 'test'
    path_both = os.path.join(args.output, PATH_BOTH)
    path_test = os.path.join(args.output, PATH_TEST)
    assert os.path.isfile(args.input_train)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(path_both, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)
    # total processing steps for tqdm progress bar
    num_steps = 5
    # reactivity column preparation
    num_reactivity_cols = 206
    reactivity_cols = [f'reactivity_{n:04}' for n in range(1, num_reactivity_cols+1)]
    with tqdm(total=num_steps) as t:
        t.set_description('loading: %s' %args.input_train)
        data = pd.read_pickle(args.input_train, compression='gzip')
        datass = pd.read_pickle(PATH_SS_TRAIN, compression='gzip')
        print('loaded:', data.shape[0])
        data.drop_duplicates(['sequence', 'experiment_type'], inplace=True)
        seqall = data['sequence'].unique().tolist()
        df_dms = data.loc[data['experiment_type'] == 'DMS_MaP'].copy()
        df_2a3 = data.loc[data['experiment_type'] == '2A3_MaP'].copy()
        df_dms['sequence'] = df_dms['sequence'].astype(str)
        seqall_df = pd.DataFrame(data=seqall, columns=['sequence'])
        seqall_df['sequence'] = seqall_df['sequence'].astype(str)
        df_dms = seqall_df.merge(df_dms, on='sequence', how='left')
        df_dms['experiment_type'] = df_dms['experiment_type'].fillna('DMS_MaP')
        df_2a3 = seqall_df.merge(df_2a3, on='sequence', how='left')
        df_2a3['experiment_type'] = df_2a3['experiment_type'].fillna('2A3_MaP')
        dfboth = pd.concat((df_dms, df_2a3))
        t.set_description('add secondary structure')
        t.update(1)

        
        dfboth = dfboth.merge(datass[['sequence', 'secondary']], on='sequence', how='left')
        assert not dfboth['secondary'].isna().any(), f'missing secondary structrure for {dfboth.secondary.isna().sum()}'
        # maintain order
        dfboth['experiment_order'] = dfboth['experiment_type'].apply(lambda x: 0 if x == 'DMS_MaP' else 1)
        dfboth.sort_values(['sequence', 'experiment_order'], inplace=True)
        assert not dfboth['experiment_type'].isna().any()
        assert dfboth.shape[0] % 2 == 0
        dfboth.to_pickle(os.path.join(path_both, 'interdata.p.gz'), compression='gzip')
        reactivity = dfboth[reactivity_cols].values
        reactivity_dms = reactivity[:, 0::2]
        reactivity_2a3 = reactivity[:, 1::2]
        reactivity_both = np.stack((reactivity_dms, reactivity_2a3), axis=-1)
        reactivity = torch.from_numpy(reactivity_both)
        reactivity = reactivity_both.clip(qmin, qmax)
        torch.save(reactivity, os.path.join(path_both, 'reactivity.pt'))
        del reactivity_2a3, reactivity_dms, reactivity_both
        ###################################################
        # filtering
        ###################################################
        # remove records with only nans
        reactivity_nan_only = ~data[reactivity_cols].isna().all(1)
        data = data.loc[reactivity_nan_only].copy()
        t.update()
        t.set_description('removing duplicates')
        records = data.shape[0]
        # data.drop_duplicates(subset='sequence', inplace=True)

        t.set_description('processing reactivity')

        # true if reactivity is not nan and seqlen
        sequences = data.sequence.tolist()[0::2]
        secondaries = data.secondary.tolist()[0::2]
        seqlen = torch.LongTensor([len(seq) for seq in sequences])
        reactivity_mask = ~torch.isnan(reactivity)
        for i in range(seqlen.shape[0]):
            reactivity_mask[i, seqlen[i]:, :] = False
        #reactivity[reactivity_mask] = 0.0
        torch.save(reactivity_mask, os.path.join(path_both, 'reactivity_mask.pt'))
        del reactivity_mask
        t.update()

        t.set_description('processing sequences')
        # sequences as ints
        sequences = [list(map(rna_map, seq)) for seq in sequences]
        secondaries = [list(map(ss_dict, ss)) for ss in secondaries]
        # seqneces to tensor
        secondaries = [torch.LongTensor(seq) for seq in sequences]
        sequences = [torch.LongTensor(seq) for seq in sequences]
        torch.save(sequences, os.path.join(path_both, 'sequences.pt'))
        torch.save(secondaries, os.path.join(path_both, 'secondary.pt'))

        sequence_id = data.sequence_id.tolist()[0::2]
        with open(os.path.join(path_both, 'sequence_id.json'), 'wt') as fp:
            json.dump(sequence_id, fp)

        del sequences
        del sequences_2a3
        del sequences_dms
        t.update()
        # processing test
        t.set_description('processing test data')
        data = pd.read_pickle(PATH_SEQ_TEST, compression='gzip')
        datass = pd.read_pickle(PATH_SS_TEST, compression='gzip')
        data = data.merge(datass[['sequence', 'secondary']], on='sequence', how='left')
        assert not data.secondary.isna().any()
        data.to_csv(os.path.join(path_test, 'test_with_secondary.csv'))

    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('converts csv file into dict with torch tensors')
    parser.add_argument('input_train', help='input .csv file')
    parser.add_argument('output', help='output directory where files will be placed')
    args = parser.parse_args()
    
    main(args)