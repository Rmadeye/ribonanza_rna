import pandas as pd
import os, gc
import numpy as np
from tqdm.notebook import tqdm
import math
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RNA_Dataset_Test(Dataset):
    def __init__(self, seq_list, **kwargs):
        # self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = max([len(seq) for seq in seq_list])
        self.seq_list = seq_list
        
    def __len__(self):
        return len(self.seq_list)  
    
    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        L = len(seq)
        mask[:L] = True
        
        # seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        seq = np.pad(seq,(0,self.Lmax-L))
        
        return {'seq':torch.from_numpy(seq), 'mask':mask}, {}
            
def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)


