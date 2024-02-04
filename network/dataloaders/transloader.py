import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class RNADatasetSecTrain(Dataset):
    # ACCCCCUUU
    # 000111000
    # ---CCC---
    seqlen_max = 206
    def __init__(self, sequences, sec_struct: list):
        # assert len(sequences) == len(masks)
        assert len(sequences) == len(sec_struct)
        # self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.sequences = sequences
        # self.masks = masks
        self.sec_struct = sec_struct

    def __len__(self):
        return len(self.sequences)  
    
    def __getitem__(self, idx) -> list:
        seq = self.sequences[idx]
        seqlen = seq.shape[0]
        ss = self.sec_struct[idx]
        mask = torch.ones(self.seqlen_max, dtype=torch.bool)
        mask[seqlen:] = False
        padsize = self.seqlen_max - seqlen
        # print(self.seqlen_max, seqlen, padsize)
        if padsize > 0:
            # pad  arg: (x, y) with value 0
            # intut 1d: [1, 2, 3, 4, 5, 6]
            # output: [0]*x + [1, 2, 3, 4, 5, 6] + [0]*y
            seq = torch.nn.functional.pad(seq, pad=(0, padsize), mode='constant', value=3)
        return seq, ss, mask
            

# class RNA_Dataset_train_both(Dataset):
#     def __init__(self, sequences, rdms: list, r2a3: list, mdms: list, m2a3: list):
#         assert len(sequences) == len(rdms)
#         assert len(sequences) == len(r2a3)
#         assert len(sequences) == len(m2a3)
#         assert len(sequences) == len(mdms)
#         # self.seq_map = {'A':0,'C':1,'G':2,'U':3}
#         self.sequences = sequences
#         self.seqlen_max = 206
#         self.m2a3 = m2a3
#         self.mdms = mdms
#         self.r2a3 = r2a3
#         self.rdms = rdms

#     def __len__(self):
#         return len(self.sequences)  
    
#     def __getitem__(self, idx) -> list:
#         seq = self.sequences[idx]
#         react_dms = self.rdms[idx]
#         react_2a3 = self.r2a3[idx]
#         mask_dms = self.mdms[idx]
#         mask_2a3 = self.m2a3[idx]
#         seqlen = seq.shape[0]
#         padsize = self.seqlen_max - seqlen
#         # [seqlen, 2]
#         mask = torch.stack((mask_dms, mask_2a3), dim=-1)
#         react = torch.stack((react_dms, react_2a3), dim=-1)
#         if padsize > 0:
#             # pad  arg: (x, y) with value 0
#             # intut 1d: [1, 2, 3, 4, 5, 6]
#             # output: [0]*x + [1, 2, 3, 4, 5, 6] + [0]*y
#             seq = torch.nn.functional.pad(seq, pad=(0, padsize), mode='constant', value=4)
#         return seq, react, seqlen, mask



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


def setup_dataloader_ss(sequences, sec_struct, device: torch.device, **kwargs):

    dataset = RNADatasetSecTrain(sequences, sec_struct)
    # __getitem__ dataset[i]
    # .call dataset()
    # __len__ len(dataset)
    dataloader = DataLoader(dataset, **kwargs)
    dataloader_on_device = DeviceDataLoader(dataloader, device)
    return dataloader_on_device



# def setup_dataloader_both(sequences,
#                            react_dms,
#                              react_2a3,
#                                mask_dms,
#                                 mask_2a3,
#                                   device: torch.device, **kwargs):

#     dataset = RNA_Dataset_train_both(sequences, react_dms, react_2a3, mask_dms, mask_2a3)
#     # __getitem__ dataset[i]
#     # .call dataset()
#     # __len__ len(dataset)
#     dataloader = DataLoader(dataset, **kwargs)
#     dataloader_on_device = DeviceDataLoader(dataloader, device)
#     return dataloader_on_device