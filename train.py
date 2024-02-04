import os
import time
import yaml
import argparse
from typing import List
from datetime import datetime
import json


import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR

from network.models.transformer_ss import SecStructModel as Network
from network.metrices import calculate_mae, calculate_mse
from network.dataloaders.transwindowerr import setup_dataloader
from network.logger import ResultsLogger


date = datetime.now().strftime("%Y%m%d-%H%M")

def prepare_yerr(yerr: torch.tensor, mask: torch.tensor):
    yerr_corrected = yerr.masked_fill_(torch.isnan(yerr), 0.9)
    yerr_corrected = 1-yerr[mask].clamp(0.01, 0.9)
    return yerr_corrected

def select_indices(tensorlist: List[torch.tensor], indexlist: List[int]):
    return [tensorlist[i] for i in indexlist]


def prepare_data(data_dir: str,
                 batch_size=32,
                 window_size=100, 
                 test: bool = False):
    global args
    
    assert os.path.isdir(data_dir)
    sequences = torch.load(os.path.join(data_dir, 'sequences.pt'))
    secondary = torch.load(os.path.join(data_dir, 'secondary.pt'))
    reactivity = torch.load(os.path.join(data_dir, 'reactivity.pt'))
    reactivity_err = torch.load(os.path.join(data_dir, 'reactivity_err.pt'))
    with open(os.path.join(data_dir, 'fold_ids.json'), 'rt') as fp:
    # with open(os.path.join(data_dir, 'folds.json'), 'rt') as fp:
        folddata = json.load(fp)
    fold_key = str(args.fold) # removed -1 for working array script
    train_index  = folddata[fold_key]['ids']
    test_index = folddata[fold_key]['ids_test']
    # train_index  = folddata[fold_key]['train_index']
    # test_index = folddata[fold_key]['test_index']
    if test:
        num_samples = 1000
        train_index = train_index[:num_samples]
        test_index = test_index[:num_samples]
    # breakpoint()
    Xtrain, Xtrainss, ytrain, ytrainerr = select_indices(sequences, train_index), select_indices(secondary, train_index), select_indices(reactivity, train_index), select_indices(reactivity_err, train_index)
    Xtrain, Xtrainss, ytrain, ytesterr = select_indices(sequences, test_index), select_indices(secondary, test_index), select_indices(reactivity, test_index), select_indices(reactivity_err, test_index)
    train_loader = setup_dataloader(sequences=Xtrain, secondary=Xtrainss, reactivities=ytrain, reactivity_err=ytrainerr,
                                    device='cuda', window_size=window_size, batch_size=batch_size)
    test_loader = setup_dataloader(sequences=Xtrain, secondary=Xtrainss, reactivities=ytrain, reactivity_err=ytesterr,
                                   device='cuda', batch_size=batch_size, window_size=window_size)
    return train_loader, test_loader


def train_model(args: argparse.Namespace):
    exp_type = args.exp_type.upper()
    data_dir = os.path.join(args.data_dir, exp_type.lower())
    model_dir = os.path.join(args.model_dir, exp_type, f"{args.fold}")

    print('using', args.hparams)
    print('device', args.device)
    print('input data:', args.data_dir)
    print('exp type: ', exp_type)
    if args.test:
        print('using test mode')
    with open(args.hparams, 'rt') as fp:
        hparams =yaml.safe_load(fp)
        network_params = hparams['network']
        train_params = hparams['train']
    data_logging = ResultsLogger(project_name='rna_kaggle',
    config=hparams,
    test_mode=args.test,
    exp_type=exp_type,
    fold=args.fold)

    device = torch.device('cuda' if args.device == 'gpu' else 'cpu')
    net = Network(**network_params).to(device)
    if args.model_dir:
        print('loading model from', model_dir)
        net.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'))['state_dict'])
    lossfn = torch.nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr=float(train_params['lr']), weight_decay=float(train_params['weight_decay']))
    scheduler = CyclicLR(optimizer, base_lr=float(train_params['lr']), max_lr=float(train_params['lr'])*10,cycle_momentum=False)
    train_loader, test_loader = prepare_data(
        data_dir, batch_size=train_params['batch_size'],
        test=args.test, window_size=network_params['window_size'])

    nsamples = len(train_loader.dataloader.dataset.sequences)
    # nsamples_test = len(test_loader.dataloader.dataset.sequences)
    best_test_error = 1e+6
    num_epochs_without_gain = 0
    epochs = 2 if args.test else train_params['epochs'] 
    # Initialize the gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        net.train()
        epoch_loss, epoch_mae, epoch_mse = 0,0,0
        epoch_time = time.perf_counter()
        true_masked = 0
        for seq, sec, y, mask, yerr in train_loader:
            # breakpoint()
            # seq, y, mask = seq.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            # Use automatic mixed precision
            with torch.cuda.amp.autocast():

                ypred = net(seq, sec, mask)
                mask_react = ~torch.isnan(y)
                yerr_adjusted = prepare_yerr(yerr, mask_react)
                loss = (lossfn(ypred[mask_react], y[mask_react])*yerr_adjusted).sum()
                # loss = (lossfn(ypred[mask_react], y[mask_react])).sum()
                loss = loss / torch.FloatTensor([seq.shape[0]]).to(seq.device)
                # breakpoint()
            # Scale the loss and call backward()
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
            true_masked += torch.count_nonzero(mask_react).item()
            epoch_loss += loss.item()
            epoch_mae += calculate_mae(y, ypred, mask_react)
            epoch_mse += calculate_mse(y, ypred, mask_react)
        
        scheduler.step(epoch_loss)
        # breakpoint()
        epoch_loss /= nsamples
        epoch_mae /= true_masked
        epoch_mse = np.sqrt(epoch_mse) / true_masked
        epoch_time = (time.perf_counter() - epoch_time)/60

        net.eval()  # set the network to evaluation mode
        with torch.no_grad():
            test_masked, test_loss, test_mae, test_mse = 0,0,0,0
            for seq, sec, y, mask, yerr in test_loader:
                ypred = net(seq, sec, mask)
                mask_react = ~torch.isnan(y)
                yerr_adjusted = prepare_yerr(yerr, mask_react)
                loss = (lossfn(ypred[mask_react], y[mask_react])*yerr_adjusted).sum()
                # loss = (lossfn(ypred[mask_react], y[mask_react])).sum()
                loss = loss / torch.FloatTensor([seq.shape[0]]).to(seq.device)
                true_masked += torch.count_nonzero(mask_react).item()
                test_loss += loss.item()
                test_mae += calculate_mae(y, ypred, mask_react)
                test_mse += calculate_mse(y, ypred, mask_react)
                test_masked += torch.count_nonzero(mask_react).item()

            test_mae /= test_masked
            test_mse = np.sqrt(test_mse) / test_masked
            test_error = test_mae
            # Checkpointing and early stopping based on test error
            if best_test_error > test_error:
                best_test_error = test_error
                if args.test:
                    pass
                else:
                    checkpoint_dir = os.path.join(args.output_dir, exp_type, f"{args.fold}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        {'state_dict' :net.state_dict(),
                        'hparams' : hparams,
                        'trainstats': {'epoch': epoch,
                                        'train_mae': epoch_mae,
                                        'test_mae': test_mae }},
                        os.path.join(checkpoint_dir, f'model.pt')
                        )                                       
                    num_epochs_without_gain = 0
            else:
                num_epochs_without_gain += 1
            if num_epochs_without_gain >= train_params['early_stopping']:
                print(f'early stopping after {epoch}')
                break

        data_logging.log(epoch,
                    epoch_loss, 
                    epoch_mae, 
                    epoch_mse, 
                    test_loss, 
                    test_mae, 
                    test_error,
                    epoch_time,
                    exp_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--hparams', type=str, help='path to hparams file', required=True)
    parser.add_argument('--device', type=str, default='cpu', help='device to train on')
    parser.add_argument('--data_dir', type=str, default='/home/nfs/kkaminski/kaggle/rna_filt', help='path to data directory')
    parser.add_argument('--output_dir', type=str, default='./data/outputs/', help='path to output directory')
    parser.add_argument('--test', action='store_true', help='run test training')
    parser.add_argument('--exp_type', choices=['2A3', 'DMS'] , help='experiment type', required=True)
    parser.add_argument('--fold', type=int, help='fold number', required=True)
    parser.add_argument('--model_dir', type=str, default=None, help='path to model directory', required=False)

    args = parser.parse_args()

    train_model(args)
