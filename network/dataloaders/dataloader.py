import torch
from torch.utils.data import DataLoader, Dataset


class RNA_Dataset_Test(Dataset):
    def __init__(self, sequences, mask = None):
        assert mask is None or len(sequences) == len(mask)
        # self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.sequences = sequences
        self.seqlen_max = max([seq.shape[0] for seq in sequences])
        if not mask is None:
            self.masks = mask
        else:
            self.masks = [torch.ones(seq.shape[0], dtype=torch.bool) for seq in sequences]
        #self.reactivity = reactivity
        
    def __len__(self):
        return len(self.masks)  
    
    def __getitem__(self, idx) -> list:
        seq = self.sequences[idx]
        mask = self.masks[idx]
        seqlen = seq.shape[0]
        padsize = self.seqlen_max - seqlen
        if padsize > 0:
            # pad  arg: (x, y) with value 0
            # intut 1d: [1, 2, 3, 4, 5, 6]
            # output: [0]*x + [1, 2, 3, 4, 5, 6] + [0]*y
            seq = torch.nn.functional.pad(seq, pad=(0, padsize), mode='constant', value=4)
            mask = torch.nn.functional.pad(mask, pad=(0, padsize), mode='constant', value=False)
        return seq, mask
            

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


def setup_dataloader(sequences, masks, device: torch.device, **kwargs):

    dataset = RNA_Dataset_Test(sequences, masks)
    # __getitem__ dataset[i]
    # .call dataset()
    # __len__ len(dataset)
    dataloader = DataLoader(dataset, **kwargs)
    dataloader_on_device = DeviceDataLoader(dataloader, device)
    return dataloader_on_device