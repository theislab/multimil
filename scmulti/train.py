import argparse
import numpy as np
import time
import os
import json
from utils import create_optimizer, parse_config_file
from datasets import load_dataset
from models import create_model
import torch


def train(experiment_name, **config):
    # config torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.manual_seed(config['train']['seed'])

    # load train and validation datasets
    train_datasets, val_datasets = load_dataset(config['dataset'], device)
    
    # create the model to be trained
    model = create_model(config['model'], device)

    # load training configurations
    n_epochs = config['train']['n-epochs']

    kl_annealing = config['train'].get('kl-annealing', None)

    early_stopping_limit = config['train'].get('early-stopping', None)
    track_best_start = 0 if kl_annealing is None else kl_annealing['start'] + kl_annealing['period']

    optimizer = create_optimizer(model.parameters(), config['train']['optimizer'])

    print(model)
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # do the training epochs
    best_model = None
    best_loss = np.inf
    early_stopping_count = 0

    for epoch in range(n_epochs):
        train_loss = 0
        train_losses = []
        epoch_time = time.time()

        # train
        model.train()
        for i, datas in enumerate(zip(*train_datasets)):
            if kl_annealing is not None:
                model.kl_anneal(epoch - kl_annealing['start'], kl_annealing['period'])
            output, loss, losses = model.forward(*datas)
            optimizer.zero_grad()
            model.backward()
            optimizer.step()

            train_loss += loss.item()
            train_losses.append(losses)

        train_losses = {k: sum([losses[k].item() for losses in train_losses]) for k in train_losses[0].keys()}
        
        epoch_time = time.time() - epoch_time

        # evaluate
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_losses = []
            for i, datas in enumerate(zip(*val_datasets)):
                loss, losses = model.test(*datas)
                val_loss += loss.item()
                val_losses.append(losses)

            val_losses = {k: sum([losses[k].item() for losses in val_losses]) for k in val_losses[0].keys()}
        
        # some descriptions to be printed
        description = []
        if epoch == track_best_start:
            description.append('tracking started')
        if kl_annealing is not None and epoch == kl_annealing['start']:
            description.append('kl annealing started')

        # keep the best model
        if epoch >= track_best_start and best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            early_stopping_count = 0
            description.append('new best')

        if epoch >= track_best_start:
            early_stopping_count += 1

        train_losses = ', '.join([f'{k}={v:.4f}' for k, v in train_losses.items()])
        val_losses = ', '.join([f'val_{k}={v:.4f}' for k, v in val_losses.items()])
        description = ', '.join(description)
        print(f'epoch {epoch+1}/{n_epochs}: time={epoch_time:.2f}(s),',
              f'loss={train_loss:.4f}, {train_losses}, val_loss={val_loss:.4f}, {val_losses}', end=' ')
        if description:
            print(f'({description})', end='')
        print()
        
        # stop the training in case of early stopping
        if early_stopping_limit is not None and early_stopping_count > early_stopping_limit:                           
            print('early stopping.')
            break
    
    # save the best model and the experiment parameters
    output_dir = os.path.join(config['train']['output-dir'], experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(best_model, os.path.join(output_dir, 'best-model.pt'))
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)
    
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