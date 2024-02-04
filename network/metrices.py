import torch


def calculate_mae(y: torch.FloatTensor, ypred: torch.FloatTensor,  mask: torch.BoolTensor) -> float:
    '''
    calculate mae over batch
    '''

    assert ypred.ndim == 2
    assert ypred.ndim == y.ndim
    assert ypred.ndim == mask.ndim
    assert ypred.shape[1] == y.shape[1]
    # breakpoint()
    mae = ypred - y
    # if mask is false override
    mae[~mask] = 0.0
    mae = mae.abs().sum()
    # torch to float
    return mae.item()


def calculate_mse(y: torch.FloatTensor, ypred: torch.FloatTensor,  mask: torch.BoolTensor) -> float:
    """
    sum Square error per batch
    """
    assert ypred.ndim == 2
    assert ypred.ndim == y.ndim
    assert ypred.ndim == mask.ndim
    assert ypred.shape[1] == y.shape[1]

    mse = ypred - y
    # if mask is false override
    mse[~mask] = 0.0
    mse = (mse**2).sum()
    return mse.item()