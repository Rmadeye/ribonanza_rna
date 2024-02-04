import os
from typing import List, Union, Tuple
from itertools import chain
from collections import defaultdict
from functools import partial
import gc

from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from network.dataloaders.transloader import DeviceDataLoader
import pandas as pd


from .models.transformer_ss import SecStructModel

def stride(iterable, window_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == window_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


# def overlap_sequence(seq: List[torch.tensor], window_size: int, stride: int) -> List[torch.tensor]:



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_collate_train(batch):
    # concatenate sequecnes into batch and pad with zeros
    (seq, react, mask) = zip(*batch)
    seqlens = [x.shape[0] for x in seq]
    max_seqlen = max(seqlens)
    # remove unnecesery reactivity values
    react = [r[:slen] for r, slen in zip(react, seqlens)]
    seqpadded = pad_sequence(seq, batch_first=True, padding_value=4)
    reactpadded = pad_sequence(react, batch_first=True, padding_value=4)
    # mask size [206]  -> [batch_size, 206]
    mask = torch.stack(mask)
    # cut to seqpadded shape [batch_size, 206] -> [batch_size, max_seqlen]
    mask = mask[:, :max_seqlen]
    return seqpadded, reactpadded, seqlens, mask

def pad_collate(batch):
    # concatenate sequecnes into batch and pad with zeros
    # print(batch[0].shape, batch.shape)

    (seq, sec) = zip(*batch)
    seqlens = [s.shape[0] for s in seq]
    # remove unnecesery reactivity values
    seqpadded = pad_sequence(seq, batch_first=True, padding_value=4)
    ss_padded = pad_sequence(sec, batch_first=True, padding_value=3)
    mask = mask_generator(seqlens, seqpadded.device)
    return seqpadded, ss_padded, mask

def mask_generator(seqlens: List[int], device: torch.device) -> torch.Tensor:
    """
    each sequence has different mask
    """
    num_sequences = len(seqlens)
    max_seqlen = max(seqlens)
    seqlens_t = torch.FloatTensor(seqlens).to(device)
    # arange [max_seqlen]
    arange = torch.arange(max_seqlen, device=device)
    # expand to [batchsize, max_seqlen]
    # [[0, 1, 2, 3, 4, ... ,200]
    # [0, 1, 2, 3, 4, ..., 201]]
    idx = arange.unsqueeze(0).expand((num_sequences, max_seqlen))
    # same with seqlens
    # [[170, 170, 170, 170]
    # [173, 173, 173, 173]]
    len_expanded = seqlens_t.unsqueeze(1).expand((num_sequences, max_seqlen))
    mask = (idx < len_expanded).bool()
    return mask

class Predictor:

    _current_index: int = 0
    _loader_params = dict(num_workers=2, pin_memory=False, shuffle=False)
    hparams = dict()
    def __init__(self,
                model_2a3_paths: list,
                model_dms_paths: list,
                output_file: str,
                window_size: int = 100,
                test: bool = False,
                device: str = 'cuda'
                 ):
        self.device = torch.device('cuda' if device in {'cuda', 'gpu'} else 'cpu')
        self.device_type = device
        self.output_file = output_file
        self.window_size = window_size
        if test:
            self.models = {'2a3_0': lambda x, y: x.float(),
                           'dms_0': lambda x, y: x.float() / 2,
                            '2a3_1': lambda x, y: x.float()+0.1,
                            'dms_1': lambda x, y: x.float() / 3,
                            '2a3_2': lambda x, y: x.float()+0.2,
                            'dms_2': lambda x, y: x.float() / 4,
                            '2a3_3': lambda x, y: x.float()+0.3,
                            'dms_3': lambda x, y: x.float() / 5,
            }
            self.hparams = {'2a3_0': {'train': {'batch_size': 32}},
                            'dms_0': {'train': {'batch_size': 32}},
                            '2a3_1': {'train': {'batch_size': 32}},
                            'dms_1': {'train': {'batch_size': 32}},
                            '2a3_2': {'train': {'batch_size': 32}},
                            'dms_2': {'train': {'batch_size': 32}},
                            '2a3_3': {'train': {'batch_size': 32}},
                            'dms_3': {'train': {'batch_size': 32}},
            }
        else:
            self.models = torch.nn.ModuleDict()
            exp_types = ['2a3', 'dms']
            model_paths = dict(zip(exp_types, [model_2a3_paths, model_dms_paths]))
            model_params_dict = {}
            

        for i in range(len(model_2a3_paths)):
            for exp_type in exp_types:
                model_name = f'{exp_type}_{i}'
                model_params_dict[model_name] = torch.load(model_paths[exp_type][i])

        for model_name, model_params in model_params_dict.items():
            trainstats = model_params['trainstats']
            hparams_str = [f"{key}: {val}" for key, val in model_params['hparams']['network'].items()]
            epoch, test_mae = trainstats['epoch'], trainstats['test_mae']
            print(f'model {model_name} from epoch: {epoch} with mae {test_mae:.2f}')
            print(" ".join(hparams_str))
            self.models[model_name] = SecStructModel(**model_params['hparams']['network'])
            self.models[model_name].load_state_dict(model_params['state_dict'])
            self.models[model_name] = self.models[model_name].eval().to(self.device)
            self.hparams[model_name] = model_params['hparams']


    def prepare_data(self, sequences, secondary) -> DataLoader:
        seqs_strided = [stride(seq, 100) for seq in sequences]
        secondaries_strided = [stride(sec, 100) for sec in secondary]
        # breakpoint()
        data = [(torch.tensor(seq), torch.tensor(ss)) for seqs, ss in zip(seqs_strided, secondaries_strided) for seq, ss in zip(seqs, ss)]
        # breakpoint()
        for _, hparams in self.hparams.items(): break
        batch_size = hparams['train']['batch_size']
        return DataLoader(dataset=data, batch_size=batch_size, collate_fn=pad_collate, **self._loader_params)

    def unpad_predictions(self, predictions, seqlens) -> List[np.array]:
        unpadded_predictions = list()
        assert len(predictions) == len(seqlens)
        for pred, seqlen in zip(predictions, seqlens):
            unpadded_pred = pred[:seqlen]
            unpadded_predictions.append(unpadded_pred)
        return unpadded_predictions
    
    def format_to_kaggle(self, results: dict) -> pd.DataFrame:
        assert isinstance(results, dict)
        assert len(results) > 0
        model_number = len(self.models)//2
# Get the values associated with the keys '2a3_i' and 'dms_i'
        values_2a3 = [results[f'2a3_{i}'] for i in range(model_number)]
        values_dms = [results[f'dms_{i}'] for i in range(model_number)]
        # Calculate the averages
        # breakpoint()
        average_2a3 = [sum(values) / len(values) for values in zip(*values_2a3)]
        average_dms = [sum(values) / len(values) for values in zip(*values_dms)]
        seqlens_2a3 = np.array([len(x) for x in results[list(results.keys())[0]]])
        seqlens_dms = np.array([len(x) for x in results[list(results.keys())[0]]])

        #assert both seqlens are identical
        assert np.all(seqlens_2a3 == seqlens_dms)

        num_reactivitis = sum(seqlens_2a3) # we need just one 

        indices = list(range(self._current_index, self._current_index + num_reactivitis))
        # flatten reactivity because Rafał can't xD
        results_flat_2a3: List[List[float]] = [r.tolist() for r in average_2a3]
        results_flat_2a3: List[float] = list(chain(*results_flat_2a3))
        results_flat_dms: List[List[float]] = [r.tolist() for r in average_dms]
        results_flat_dms: List[float] = list(chain(*results_flat_dms))

        data = list(zip(indices, results_flat_dms,results_flat_2a3))
        # TODO add col names to class variables after class definition before init
        df = pd.DataFrame(data, columns=['id','reactivity_DMS_MaP','reactivity_2A3_MaP'])
        # self._current_index += num_reactivitis
        return df, num_reactivitis
    
    
    def save_data_in_chunks(self, data: tuple) -> None:
        
        header = True if self._current_index == 0 else False
        df, reactivities = data
        df['id'] = df['id'].astype(int)
        df['reactivity_2A3_MaP'] = df['reactivity_2A3_MaP'].round(3)
        df['reactivity_DMS_MaP'] = df['reactivity_DMS_MaP'].round(3)
        if header:
            df.to_csv(self.output_file, mode='w', header=True, index=False)
        else:
            df.to_csv(self.output_file, mode = 'a', header=False, index=False)
        self._current_index += reactivities
        gc.collect()


    def predict(self, sequences, secondary, crude_predictions: bool = False) -> List[np.array]:

        dataloader = self.prepare_data(sequences=sequences, secondary=secondary)
        predict_loader = DeviceDataLoader(dataloader, self.device)
        print(f"VRAM memory allocated before prediction {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
        print(f"VRAM memory cached before prediction {torch.cuda.memory_cached()/(1024**3):.2f} GB")
        with torch.no_grad(), torch.cuda.amp.autocast():
            for seq, secondary, mask in tqdm(predict_loader):
                result_unpadded = dict()
                for name, model in self.models.items():
                    breakpoint()
                    ypred_all= torch.zeros_like(seq)
                    counts = torch.zeros_like(seq)
                    for i in range(0,len(seq),10):
                        window = seq[i:i+self.window_size]
                        ypred_partial = model(window, secondary[i:i+self.window_size], mask[i:i+self.window_size])
                        ypred_all[i:i+self.window_size] += ypred_partial
                        counts[i:i+self.window_size] += 1
                    
                    ypred = ypred_all/counts
                    # ypred = model(seq, secondary, mask)
                    ypred = ypred.detach().cpu()
                    ypred = ypred.squeeze(0).numpy()
                    result_unpadded[name] = self.unpad_predictions(ypred, mask.sum(1).tolist())

                if not crude_predictions:
                    self.save_data_in_chunks(self.format_to_kaggle(result_unpadded))
                # else:
                #     print("bo ja tak mówię: ")
                #     [results[name].append(result_unpadded[name]) for name in result_unpadded.keys()]
                #     return results
