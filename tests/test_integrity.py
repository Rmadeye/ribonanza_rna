import pytest
import torch
import numpy as np
import pandas as pd
import os
from itertools import chain

from network.base import Predictor


RESULT_CSV = 'tmp.csv'
# Define a fixture that returns your sequences
@pytest.fixture(autouse=True)
def remove_cache():
    if os.path.isfile(RESULT_CSV):
        os.remove(RESULT_CSV)

@pytest.fixture
def sequences():
    torch.manual_seed(2137)
    seqlens = torch.randint(100, 400, size=(1000, ))
    seqs = [torch.randint(1, 4, size=(seqlen,)) for seqlen in seqlens]
    return seqs


@pytest.mark.parametrize('output_file', ['tmp.csv'])
def test_prediction_kaggle_df_chunks(output_file: str, sequences) -> None:
    model =  Predictor(model_2a3_paths='',model_dms_paths='',output_file=output_file, test = True)
    unpack = torch.cat(sequences).view(-1, 1)
    output_length = unpack.shape[0]
    # breakpoint()
    model.predict(sequences)
    df = pd.read_csv(output_file)
    assert list(df.columns.values) == ['id', 'reactivity_2A3_MaP', 'reactivity_DMS_MaP']
    assert df.shape[0] == output_length, f'Output length does not match input length (output_shape: {df.shape[0]}, input_shape: {output_length})'

def test_prediction_crude_data(sequences):
    model =  Predictor(model_2a3_paths='',model_dms_paths='', output_file='',test = True)
    pred = model.predict(sequences, crude_predictions=True)
    # check type
    assert isinstance(pred, dict)
    assert len(pred) > 0
    flat_pred_2a3 = list(chain(*pred['2a3']))
    flat_pred_dms = list(chain(*pred['dms']))
    # breakpoint()
    # check if number of sequences is correct
    assert len(sequences) == len(flat_pred_2a3) and len(sequences) == len(flat_pred_dms)
    num_sequences = len(sequences)
    for i in range(num_sequences):
        assert len(flat_pred_2a3[i]) == sequences[i].shape[0]
        assert len(flat_pred_dms[i]) == sequences[i].shape[0]