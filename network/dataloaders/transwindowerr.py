from typing import List

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    # ACCCCCUUU
    # 000111000
    # ---CCC---
    # A- token
    # A( token
    seqlen_max = 206
    non_nans = list()
    def __init__(self, sequences, secondary, reactivtities: list, reactivitities_err: list, window_size: int):
        # assert len(sequences) == len(masks)
        assert isinstance(window_size, int)
        assert 0 < window_size < self.seqlen_max
        assert len(sequences) == len(reactivtities)
        assert len(sequences) == len(secondary)
        self.window_size = window_size
        # self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.sequences = sequences
        self.seqlens = torch.LongTensor([seq.shape[0] for seq in self.sequences])
        assert (self.seqlens >= self.window_size).all()
        self.secondary = secondary
        # self.masks = masks
        self.reactivtities = reactivtities
        self.reactivtities_err = reactivitities_err
        #self.nans: List[torch.BoolTensor] = [torch.zeros_like(seq, dtype=torch.bool) for seq in self.sequences]
        self._create_nanmask()

    def __len__(self):
        return len(self.sequences)  
    
    def __getitem__(self, idx) -> list:
        seq = self.sequences[idx]
        sec = self.secondary[idx]
        reactivity = self.reactivtities[idx]
        seqlen = self.seqlens[idx]
        reactivity_err = self.reactivtities_err[idx]

        stride = seqlen - self.window_size # >= 0
        # modstride 0 < modstride <= stride - lowest number of nans
        nonnan_start = self.non_nans[idx][:self.window_size].sum()
        nonnan_stop = self.non_nans[idx][stride:self.window_size+stride].sum()
        is_start_better = nonnan_start > nonnan_stop
        if is_start_better: # lower stride
            low_stride = 0
            high_stride = stride//2
        else: # higher stride
            low_stride = stride//2
            high_stride = stride
        # generate stride
        stride = torch.randint(low=low_stride, high=high_stride, size=(1,))
        seq = seq[stride:stride+self.window_size]
        sec = sec[stride:stride+self.window_size]
        reactivity = reactivity[stride:stride+self.window_size]
        reactivity_err = reactivity_err[stride:stride+self.window_size]
        mask = torch.ones_like(seq, dtype=torch.bool)
        return seq, sec, reactivity, mask, reactivity_err

    def _create_nanmask(self):

        self.non_nans = [~torch.isnan(react) for react in self.reactivtities]
        # self.non_nans_err = [~torch.isnan(reacterr) for reacterr in self.reactivtities_err]
        for i, seqlen in enumerate(self.seqlens):
            self.non_nans[i][seqlen:] = False
            # self.non_nans_err[i][seqlen:] = False


def list_to_device(x, device: torch.device):
    return [el.to(device) for el in x]


class DeviceDataLoader:
    def __init__(self, dataloader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield list_to_device(batch, self.device)


def setup_dataloader(sequences, secondary, reactivities, reactivity_err, device: torch.device, window_size, **kwargs):

    dataset = WindowDataset(sequences, secondary, reactivities, reactivity_err, window_size=window_size)
    # __getitem__ dataset[i]
    # .call dataset()
    # __len__ len(dataset)
    dataloader = DataLoader(dataset, **kwargs)
    dataloader_on_device = DeviceDataLoader(dataloader, device)
    return dataloader_on_device
