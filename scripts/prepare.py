import os
import time
from typing import List
import argparse
from tqdm import tqdm
import shutil

import pandas as pd
import numpy as np
import torch

def mask_and_save(source, mask, file):

    source_masked = source[mask]
    torch.save(source_masked, file)


def main(args: argparse.Namespace):

    # params
    # clip reactivity to range (-q99, q99)
    # q99 = 1.1
    # q99 should be between 0 and 1
    qmin = 0.0
    qmax = 1.0

    PATH_DMS = 'dms'
    PATH_2A3 = '2a3'
    path_dms = os.path.join(args.output, PATH_DMS)
    path_2a3 = os.path.join(args.output, PATH_2A3)
    assert os.path.isfile(args.input)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(path_dms, exist_ok=True)
    os.makedirs(path_2a3, exist_ok=True)
    data = pd.read_csv(args.input, nrows=5)
    # total processing steps for tqdm progress bar
    num_steps = 6
    # reactivity column preparation
    num_reactivity_cols = 206
    reactivity_cols_all = [col for col in data.columns if col.startswith('reactivity_')]
    reactivity_cols = reactivity_cols_all[:num_reactivity_cols]
    reactivity_err_cols = reactivity_cols_all[num_reactivity_cols:]
    reactivity_dtype = {colname: np.float32 for colname in reactivity_cols_all}
    with tqdm(total=num_steps) as t:
        t.set_description('loading: %s' %args.input)
        data = pd.read_csv(args.input, dtype=reactivity_dtype, engine="c")
        dfdms = data.loc[data['experiment_type'] == 'DMS_MaP']
        df2a3 = data.loc[data['experiment_type'] == '2A3_MaP']
        data = pd.concat([dfdms.drop_duplicates(subset='sequence'), df2a3.drop_duplicates(subset='sequence')], axis=0)
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
        # decode 
        rna_dict = dict(A=0, C=1, G=2, U=3)
        rna_map = lambda letter : rna_dict[letter]
        t.set_description('processing reactivity')

        reactivity = data[reactivity_cols].values.astype(np.float32)
        reactivity_mask = ~np.isnan(reactivity)
        #reactivity[reactivity_mask] = 0.0
        reactivity = torch.from_numpy(reactivity).float()
        reactivity = reactivity.clip(qmin, qmax)
        if (reactivity[~reactivity.isnan()] == 0.0).all():
            raise ValueError('reactivity is always zero')
        reactivity_dms = reactivity[experiment_mask, :]
        torch.save(reactivity_dms, os.path.join(path_dms, f'reactivity.pt'))
        del reactivity_dms
        reactivity_2a3 = reactivity[~experiment_mask, :]
        torch.save(reactivity_2a3, os.path.join(path_2a3, f'reactivity.pt'))
        del reactivity_2a3
        del reactivity
        t.update()

        reactivity_mask = torch.from_numpy(reactivity_mask)
        reactivity_mask_dms = reactivity_mask[experiment_mask, :]
        torch.save(reactivity_mask_dms, os.path.join(path_dms, 'reactivity_mask.pt'))
        del reactivity_mask_dms
        reactivity_mask_2a3 = reactivity_mask[~experiment_mask, :]
        torch.save(reactivity_mask_2a3, os.path.join(path_2a3, 'reactivity_mask.pt'))
        del reactivity_mask_2a3
        del reactivity_mask
        t.update()

        reactivity_err = data[reactivity_err_cols].values.astype(np.float32)
        reactivity_err = torch.from_numpy(reactivity_err)
        reactivity_err_dms = reactivity_err[experiment_mask, :]
        torch.save(reactivity_err_dms, os.path.join(path_dms, 'reactivity_err.pt'))
        del reactivity_err_dms
        reactivity_err_2a3 = reactivity_err[~experiment_mask, :]
        torch.save(reactivity_err_2a3, os.path.join(path_2a3, 'reactivity_err.pt'))
        del reactivity_err_2a3
        del reactivity_err
        t.update()

        t.set_description('processing sequences')
        sequences = data['sequence'].tolist()
        sequences_len = [list(map(len, seq)) for seq in sequences]
        # sequences as ints
        sequences = [list(map(rna_map, seq)) for seq in sequences]
        # seqneces to tensor
        sequences = [torch.LongTensor(seq) for seq in sequences]
        sequences_dms = [seq for mask, seq in zip(experiment_type, sequences) if mask]
        torch.save(sequences_dms, os.path.join(path_dms, 'sequences.pt'))
        sequences_2a3 = [seq for mask, seq in zip(experiment_type, sequences) if not mask]
        torch.save(sequences_2a3, os.path.join(path_2a3, 'sequences.pt'))
        t.update()
    print('total records %d' % records)
    print('removed %d duplicates' % records_after)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('converts csv file into dict with torch tensors')
    parser.add_argument('input', help='input .csv file')
    parser.add_argument('output', help='output directory where files will be placed')
    args = parser.parse_args()
    
    main(args)