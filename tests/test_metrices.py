import pytest
import torch
import numpy as np

from network.metrices import calculate_mae, calculate_mse

@pytest.fixture
def dataset():
    torch.manual_seed(2137)
    ypred = torch.randn(1000,2)
    y = torch.randn(1000,2)
    mask = torch.ones(1000,2)
    mask[torch.rand(1000,2) > 0.5] = 0
    mask = mask.bool()
    return  y,ypred,mask


def test_calculate_mae(dataset):
    mae = calculate_mae(*dataset)
    assert isinstance(mae, float)
    assert mae > 0.0

def test_calculate_mse(dataset):
    mse = calculate_mse(*dataset)
    assert isinstance(mse, float)
    assert mse > 0.0