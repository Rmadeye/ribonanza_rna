import os
import time
from typing import List
import argparse
from tqdm import tqdm
import shutil

import pandas as pd
import numpy as np
import torch

from data_version import (PATH_SEQ_TEST,
                          PATH_SEQ_TRAIN,
                          PATH_SS_TEST,
                          PATH_SS_TRAIN)


from tools import seqlist_to_fasta_str

rna_dict = dict(A=0, C=1, G=2, U=3)
rna_map = lambda letter : rna_dict[letter]
ss_dict = {'(': 0, ')' : 1, '.': 2}
ss_map = lambda letter : ss_dict[letter]




def main(args: argparse.Namespace):

    # params
    # clip reactivity to range (-q99, q99)
    # q99 = 1.1
    # q99 should be between 0 and 1
    qmin = 0.0
    qmax = 1.0
    signal_to_noise = 0.8

    PATH_DMS = 'dms'
    PATH_2A3 = '2a3'
    PATH_TEST = 'test'

    suffix = f"{int(signal_to_noise*100)}"
    path_dms = os.path.join(args.output + suffix, PATH_DMS)
    path_2a3 = os.path.join(args.output + suffix, PATH_2A3)
    path_test = os.path.join(args.output + suffix, PATH_TEST)
    signal_to_noise = args.sn
    
    assert os.path.isfile(PATH_SEQ_TRAIN)
    assert os.path.isfile(PATH_SS_TRAIN)
    assert os.path.isfile(PATH_SEQ_TEST)
    assert os.path.isfile(PATH_SS_TEST)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(path_dms, exist_ok=True)
    os.makedirs(path_2a3, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)
    # total processing steps for tqdm progress bar
    num_steps = 8
    # reactivity column preparation
    num_reactivity_cols = 206
    reactivity_cols = [f'reactivity_{n:04}' for n in range(1, num_reactivity_cols+1)]
    reactivity_err_cols = [f'reactivity_error_{n:04}' for n in range(1, num_reactivity_cols+1)]
    with tqdm(total=num_steps) as t:
        t.set_description('loading: %s' % PATH_SEQ_TRAIN)
        data = pd.read_pickle(PATH_SEQ_TRAIN, compression='gzip')
        t.set_description('loadingL %s' % PATH_SS_TRAIN)
        datass = pd.read_pickle(PATH_SS_TRAIN, compression='gzip')
        # data = data[data.SN_filter == 1].copy()
        data.drop_duplicates(['sequence', 'experiment_type'], inplace=True)
        sn_drop = data.shape[0]
        data = data[data.signal_to_noise <= signal_to_noise].copy()
        sn_drop -= data.shape[0]
        data = data.merge(datass[['sequence', 'secondary']], on='sequence', how='left')
        assert not data.secondary.isna().any()
        data.sort_values(['experiment_type', 'sequence'], inplace=True)
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
        experiment_type = data['experiment_type'].tolist()
        experiment_type: List[bool] = [True if exp_type == 'DMS_MaP' else False for exp_type in experiment_type]
        # true for DMS
        experiment_mask = torch.BoolTensor(experiment_type)
        records_after = records - data.shape[0]
        t.update()
        t.set_description('processing %d unique records from %d total' % (records_after, records))

        t.set_description('processing reactivity')

        reactivity = data[reactivity_cols].values.astype(np.float32)
        reactivity = torch.from_numpy(reactivity).float()

        # true if reactivity is not nan and seqlen
        seqlen = torch.LongTensor([len(seq) for seq in data.sequence.tolist()])
        reactivity_mask = ~torch.isnan(reactivity)
        for i in range(seqlen.shape[0]):
            reactivity_mask[i, seqlen[i]:] = False
        #reactivity[reactivity_mask] = 0.0
        reactivity = reactivity.clip(qmin, qmax)
        #if (reactivity[~reactivity.isnan()] == 0.0).all():
        #    raise ValueError(f'reactivity is always zero {reactivity[~reactivity.isnan()].mean() }')
        reactivity_dms = reactivity[experiment_mask, :]
        torch.save(reactivity_dms, os.path.join(path_dms, f'reactivity.pt'))
        del reactivity_dms
        reactivity_2a3 = reactivity[~experiment_mask, :]
        torch.save(reactivity_2a3, os.path.join(path_2a3, f'reactivity.pt'))
        del reactivity_2a3
        del reactivity
        t.update()


        reactivity_mask_dms = reactivity_mask[experiment_mask, :]
        torch.save(reactivity_mask_dms, os.path.join(path_dms, 'reactivity_mask.pt'))
        del reactivity_mask_dms
        reactivity_mask_2a3 = reactivity_mask[~experiment_mask, :]
        torch.save(reactivity_mask_2a3, os.path.join(path_2a3, 'reactivity_mask.pt'))
        del reactivity_mask_2a3
        del reactivity_mask
        t.update()
        t.set_description('reactivity errors')
        reactivity_err = data[reactivity_err_cols].values
        reactivity_err = torch.from_numpy(reactivity_err)

        reactivity_err_dms = reactivity_err[experiment_mask, :]
        torch.save(reactivity_err_dms, os.path.join(path_dms, 'reactivity_err.pt'))
        del reactivity_err_dms
        reactivity_err_2a3 = reactivity_err[~experiment_mask, :]
        torch.save(reactivity_err_2a3, os.path.join(path_2a3, 'reactivity_err.pt'))
        del reactivity_err_2a3



        t.set_description('processing sequences')
        sequences = data['sequence'].tolist()
        secondary = data['secondary'].tolist()
        sequences_str = data['sequence'].tolist()
        sequences_id = data['sequence_id'].tolist()
        # sequences as ints
        sequences = [list(map(rna_map, seq)) for seq in sequences]
        secondary = [list(map(ss_map, ss)) for ss in secondary]
        # seqneces to tensor
        sequences = [torch.LongTensor(seq) for seq in sequences]
        secondary = [torch.LongTensor(ss) for ss in secondary]

        sequences_dms = [seq for mask, seq in zip(experiment_type, sequences) if mask]
        sequences_dms_str = [seq for mask, seq in zip(experiment_type, sequences_str) if mask]
        sequences_dms_id = [seqid for mask, seqid in zip(experiment_type, sequences_id) if mask]
        secondary_dms = [ss for mask, ss in zip(experiment_type, secondary) if mask]
        torch.save(sequences_dms, os.path.join(path_dms, 'sequences.pt'))
        torch.save(secondary_dms, os.path.join(path_dms, 'secondary.pt'))
        fasta_str = seqlist_to_fasta_str(sequences_dms_id, sequences_dms_str)
        with open(os.path.join(path_dms, 'sequences.fasta'), 'wt') as fp:
            fp.write(fasta_str)
        dms_df = pd.DataFrame(zip(sequences_dms_id, sequences_dms_str), columns=['sequence_id', 'sequence'])
        dms_df.to_csv(os.path.join(path_dms, 'sequence_id.csv'))
        
        sequences_2a3 = [seq for mask, seq in zip(experiment_type, sequences) if not mask]
        sequences_2a3_str = [seq for mask, seq in zip(experiment_type, sequences_str) if not mask]
        sequences_2a3_id = [seqid for mask, seqid in zip(experiment_type, sequences_id) if not mask]
        secondary_2a3 = [ss for mask, ss in zip(experiment_type, secondary) if not mask]
        torch.save(sequences_2a3, os.path.join(path_2a3, 'sequences.pt'))
        torch.save(secondary_2a3, os.path.join(path_2a3, 'secondary.pt'))
        fasta_str = seqlist_to_fasta_str(sequences_2a3_id, sequences_2a3_str)
        with open(os.path.join(path_2a3, 'sequences.fasta'), 'wt') as fp:
            fp.write(fasta_str)
        df_2a3 = pd.DataFrame(zip(sequences_2a3_id, sequences_2a3_str), columns=['sequence_id', 'sequence'])
        df_2a3.to_csv(os.path.join(path_2a3, 'sequence_id.csv'))

        del sequences
        del secondary
        del sequences_2a3
        del sequences_dms
        del secondary_2a3
        del secondary_dms
        t.update()

        t.set_description('processing test seq')
        # test data
        data = pd.read_pickle(PATH_SEQ_TEST, compression='gzip')
        datass = pd.read_pickle(PATH_SS_TEST, compression='gzip')
        t.update()
        data = data.merge(datass[['sequence', 'secondary']], on='sequence', how='left')
        assert not data.secondary.isna().any()
        data.to_csv(os.path.join(path_test, 'test_with_secondary.csv'))
        t.update()

        
    print('total records %d' % records)
    print('signal to noise excluded: ', sn_drop)
    print('removed %d duplicates' % records_after)
    print('removed %d rows with nans' % reactivity_nan_only.sum())
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('converts csv file into dict with torch tensors')
    parser.add_argument('output', help='output directory where files will be placed')
    parser.add_argument('--sn', type=float, default=1.0)
    args = parser.parse_args()
    
    main(args)
