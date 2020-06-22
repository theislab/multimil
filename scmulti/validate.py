import argparse
import numpy as np
import time
import os
import json
from utils import create_optimizer, parse_config_file, plot_latent
from datasets import load_dataset
from models import load_model
import torch


def validate(**config):
    # config torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['train']['seed'])

    # load train and validation datasets
    train_datasets, val_datasets = load_dataset(config['dataset'], device)
    
    # get configs
    output_dir = os.path.join(config['train']['output-dir'], config['experiment-name'])

    # create the model to be trained
    model_path = os.path.join(output_dir, 'best-model.pt')
    model = load_model(config['model'], model_path, device)

    # calculate validation loss
    # find integration of data
    with torch.no_grad():
        model.eval()

        # validate on training data
        train_loss = 0
        for i, datas in enumerate(zip(*train_datasets)):
            train_loss += model.test(*datas).item()

        # validate on validation data
        val_loss = 0
        for i, datas in enumerate(zip(*val_datasets)):
            val_loss += model.test(*datas).item()

        print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        # centers at which datasets should gather togher
        centers = [None] + list(range(len(train_datasets)))

        # integrate training datasets into common latent space
        zs_train = {}
        for center in centers:  # integrate datasets into multiple centers
            zs_train[center] = []
            for i, datas in enumerate(zip(*train_datasets)):
                z = model.integrate(*datas)
                zs_train[center].append(torch.stack(z))
            zs_train[center] = torch.cat(zs_train[center], dim=1)

        # integrate validation datasets into common latent space
        zs_val = {}
        for center in centers: # integrate datasets into multiple centers
            zs_val[center] = []
            for i, datas in enumerate(zip(*val_datasets)):
                z = model.integrate(*datas)
                zs_val[center].append(torch.stack(z))
            zs_val[center] = torch.cat(zs_val[center], dim=1)

        for center in centers:
            plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in train_datasets],
                        zs_train[center].cpu().numpy(), output_dir, f'train-at-{center}-')
            plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in val_datasets],
                        zs_val[center].cpu().numpy(), output_dir, f'val-at-{center}-')


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('--root-dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(os.path.join(args.root_dir, 'config.json'))

    validate(**config)