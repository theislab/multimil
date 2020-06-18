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
        zs_train = []
        for i, datas in enumerate(zip(*train_datasets)):
            train_loss += model.test(*datas).item()
            z = model.integrate(*datas)
            zs_train.append(torch.stack(model.integrate(*datas)))
        zs_train = torch.cat(zs_train, dim=1)
        train_loss /= len(train_datasets[0])

        # validate on validation data
        val_loss = 0
        zs_val = []
        for i, datas in enumerate(zip(*val_datasets)):
            val_loss += model.test(*datas).item()
            z = model.integrate(*datas)
            zs_val.append(torch.stack(model.integrate(*datas)))
        zs_val = torch.cat(zs_val, dim=1)
        val_loss /= len(val_datasets[0])

    print(f'train_loss={train_loss:.3f}, val_loss={val_loss:.3f}')

    plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in train_datasets], zs_train.cpu().numpy(), output_dir, 'train-')
    plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in val_datasets], zs_val.cpu().numpy(), output_dir, 'val-')


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('--root-dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(os.path.join(args.root_dir, 'config.json'))

    validate(**config)