import argparse
import numpy as np
import time
import os
import json
from itertools import cycle
from utils import create_optimizer, parse_config_file
from datasets import load_dataset
from models import create_model
import torch


def train(experiment_name, **config):
    # config torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.manual_seed(config['train']['seed'])

    # configure output directory to save logs and results
    output_dir = os.path.join(config['train']['output-dir'], experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'train.txt')
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)

    # load train and validation dataloaders
    train_dataloder, val_dataloader = load_dataset(config['dataset'], device)
    
    # create the model to be trained
    model = create_model(config['model'], device)

    # load training configurations
    n_epochs = config['train']['n-epochs']
    warmup = config['train'].get('warmup', 0)
    kl_annealing = config['train'].get('kl-annealing', None)
    early_stopping_limit = config['train'].get('early-stopping', None)
    start_tracking = warmup if kl_annealing is None else warmup + kl_annealing
    print_every = config['train'].get('print-every', 50)
    ae_optimize_every = config['train'].get('ae-optimize-every', 3)

    optimizer_ae = create_optimizer(model.get_nonadversarial_params(), config['train']['optimizer'])
    optimizer_adv = create_optimizer(model.get_adversarial_params(), config['train']['optimizer'])

    with open(log_path, 'w') as log_file:
        print(model, file=log_file)
        print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad), file=log_file)
        print(f'device = {device}', file=log_file)

    # do the training epochs
    track_metric = config['train']['track-metric']
    best_metric = np.inf
    best_model = None
    early_stopping_count = 0

    # train
    for epoch, (xs, pair_indices) in enumerate(cycle(train_dataloder)):
        model.train()
        if epoch >= n_epochs:
            break

        train_loss = 0
        train_losses = []
        epoch_time = time.time()

        if epoch < warmup:
            model.warmup_mode(True)
        else:
            model.warmup_mode(False)
            if kl_annealing is not None:
                model.kl_anneal(epoch - warmup, kl_annealing)

        output, loss, losses = model.forward(xs, pair_indices)
        optimizer_ae.zero_grad()
        optimizer_adv.zero_grad()
        if epoch < warmup or epoch % ae_optimize_every == 0:
            model.backward()
            optimizer_ae.step()
        else:
            model.backward_adv()
            optimizer_adv.step()

        train_loss += loss.item()
        train_losses.append(losses)

        if epoch % print_every == 0:
            train_losses = {k: sum([losses[k].item() for losses in train_losses]) for k in train_losses[0].keys()}
            epoch_time = time.time() - epoch_time

            # evaluate
            with torch.no_grad():
                model.eval()
                val_loss = 0
                val_losses = []
                for val_xs, val_pair_indices in val_dataloader:
                    loss, losses = model.test(val_xs, val_pair_indices)
                    val_loss += loss.item()
                    val_losses.append(losses)
                val_losses = {k: sum([losses[k].item() for losses in val_losses]) for k in val_losses[0].keys()}
        
            # some descriptions to be printed
            description = []
            if epoch >= start_tracking and epoch - print_every < start_tracking:
                description.append('tracking started')
            if kl_annealing is not None and epoch >= warmup and epoch - print_every < warmup:
                description.append('kl annealing started')

            # keep the best model
            val_metric = val_loss if track_metric == 'loss' else val_losses[track_metric]
            if epoch >= start_tracking and best_metric > val_metric:
                best_metric = val_metric
                best_model = model.state_dict()
                torch.save(best_model, os.path.join(output_dir, 'best-model.pt'))  # save the best model so far
                early_stopping_count = 0
                description.append('new best')

            if epoch >= start_tracking:
                early_stopping_count += print_every

            train_losses = ', '.join([f'{k}={v:.4f}' for k, v in train_losses.items()])
            val_losses = ', '.join([f'val_{k}={v:.4f}' for k, v in val_losses.items()])
            description = ', '.join(description)

            with open(log_path, 'a') as log_file:
                print(f'epoch {epoch+1}/{n_epochs}: time={epoch_time:.2f}(s),',
                    f'loss={train_loss:.4f}, {train_losses}, val_loss={val_loss:.4f}, {val_losses}', end=' ', file=log_file)
                if description:
                    print(f'({description})', end='', file=log_file)
                print(file=log_file)
            
            # stop the training in case of early stopping
            if early_stopping_limit is not None and early_stopping_count > early_stopping_limit:                           
                with open(log_path, 'a') as log_file:
                    print('early stopping.', file=log_file)
                break

    # save the last state of the model
    last_model = model.state_dict()
    torch.save(last_model, os.path.join(output_dir, 'last-model.pt'))
    
    return best_model


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--config-file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(args.config_file)
    experiment_name = os.path.splitext(os.path.basename(args.config_file))[0]

    train(experiment_name, **config)