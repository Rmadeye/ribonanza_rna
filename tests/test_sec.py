import pytest
import torch
import numpy as np

from network.models.transformer_ss import SecStructModel

@pytest.fixture
def dataset():
    torch.manual_seed(2137)
    seq = torch.randint(0,4,(1000,100))
    torch.manual_seed(66)
    ss = torch.randint(0,2,(1000,100))
    mask = torch.ones(1000,100)
    mask[torch.rand(1000,100) > 0.5] = 0
    mask = mask.bool()
    return seq,ss,mask

def test_ss_model(dataset):
    model = SecStructModel()
    seq,ss,mask = dataset
    y = model(seq,ss,mask)
    breakpoint()
    assert y.shape == (1000,100)