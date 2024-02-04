import os
import argparse

import time
import pandas as pd
import torch

from network.transwinpred_overlap import Predictor
from scripts.tools import rna_map, ss_map


if __name__ == '__main__':
    chunksize = 256*10
    testdata= '/path/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=False)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fold', required=False, type=int)
    args = parser.parse_args()
    if args.test:
        model_dms_paths = None
        model_2a3_paths = None
    else:
        if args.fold is None:
            folds = list(range(0,4))            
            model_2a3_paths=[os.path.join(args.model_dir, '2A3', str(fold), 'model.pt') for fold in folds]
            model_dms_paths=[os.path.join(args.model_dir, 'DMS', str(fold), 'model.pt') for fold in folds]
        else:
            model_2a3_paths=[os.path.join(args.model_dir, '2A3', str(args.fold), 'model.pt')]
            model_dms_paths=[os.path.join(args.model_dir, 'DMS', str(args.fold), 'model.pt')]
    predictor = Predictor(model_2a3_paths=model_2a3_paths,
                          model_dms_paths=model_dms_paths,
                           output_file=args.output_file,
                           )
    print(f'loading file: {testdata}')
    assert os.path.isfile(testdata)
    if os.path.isfile(args.output_file):
        print('outfile found it will be removed')
        os.remove(args.output_file)
    data = pd.read_csv(testdata, chunksize=chunksize)
    t0 = time.perf_counter()
    for chunk in data:
        sequences = [list(map(rna_map, seq)) for seq in chunk.sequence.tolist()]
        sequences = [torch.LongTensor(seq) for seq in sequences]
        secondary = [list(map(ss_map, sec)) for sec in chunk.secondary.tolist()]
        secondary = [torch.LongTensor(sec) for sec in secondary]
        chunk_data = (sequences, secondary)
        predictor.predict(sequences, secondary)
    time_total = (time.perf_counter() - t0)/60
    print(f'Done in {time_total:.2f} min')
