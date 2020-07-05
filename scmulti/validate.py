import argparse
import numpy as np
import pandas as pd
import time
import os
import json
from utils import create_optimizer, parse_config_file, plot_latent
from metrics import entropy_batch_mixing, knn_purity
from datasets import load_dataset
from models import load_model
import scanpy as sc
import torch


def validate(experiment_name, **config):
    # config torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['train']['seed'])

    # load train and validation datasets
    train_datasets, val_datasets = load_dataset(config['dataset'], device)
    
    # get configs
    output_dir = os.path.join(config['train']['output-dir'], experiment_name)
    log_file = open(os.path.join(output_dir, 'evaluations.txt'), 'w')

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
            loss, losses = model.test(*datas)
            train_loss += loss.item()

        # validate on validation data
        val_loss = 0
        for i, datas in enumerate(zip(*val_datasets)):
            loss, losses = model.test(*datas)
            val_loss += loss.item()

        print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f}', file=log_file)

        # bring training datasets into latent space without integration
        zs_train_uninteg = []
        for i, datas in enumerate(zip(*train_datasets)):
            zs = [model.to_latent(d, j) for j, d in enumerate(datas)]
            zs_train_uninteg.append(torch.stack(zs))
        zs_train_uninteg = torch.cat(zs_train_uninteg, dim=1)

        # bring validation datasets into latent space without integration
        zs_val_uninteg = []
        for i, datas in enumerate(zip(*val_datasets)):
            zs = [model.to_latent(d, j) for j, d in enumerate(datas)]
            zs_val_uninteg.append(torch.stack(zs))
        zs_val_uninteg = torch.cat(zs_val_uninteg, dim=1)

        # calculate metrics for unintegrated latents
        train_metrics_uninteg = calc_metrics([d.dataset.dataset.get_adata()[d.dataset.indices] for d in train_datasets],
                                             zs_train_uninteg.cpu().numpy())
        val_metrics_uninteg = calc_metrics([d.dataset.dataset.get_adata()[d.dataset.indices] for d in val_datasets],
                                           zs_val_uninteg.cpu().numpy())
        train_metrics_uninteg = ', '.join([f'{k}={v:.4f}' for k, v in train_metrics_uninteg.items()])
        val_metrics_uninteg = ', '.join([f'{k}={v:.4f}' for k, v in val_metrics_uninteg.items()])
        print('metrics for unintegrated training set:', train_metrics_uninteg, file=log_file)
        print('metrics for unintegrated validation set:', val_metrics_uninteg, file=log_file)

        # plot the unintegrated latents 
        plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in train_datasets],
                    zs_train_uninteg.cpu().numpy(), output_dir, 'train-uintegrated-')
        plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in val_datasets],
                    zs_val_uninteg.cpu().numpy(), output_dir, 'val-unintegrated-')

        ### INTEGRATION ###
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

        # evaluate integration
        for center in centers:
            # calculate metrics for integrated latents
            train_metrics = calc_metrics([d.dataset.dataset.get_adata()[d.dataset.indices] for d in train_datasets],
                                         zs_train[center].cpu().numpy())
            val_metrics = calc_metrics([d.dataset.dataset.get_adata()[d.dataset.indices] for d in val_datasets],
                                       zs_val[center].cpu().numpy())
            train_metrics = ', '.join([f'{k}={v:.4f}' for k, v in train_metrics.items()])
            val_metrics = ', '.join([f'{k}={v:.4f}' for k, v in val_metrics.items()])
            print(f'metrics for integrated training set at center {center}:', train_metrics, file=log_file)
            print(f'metrics for integrated validation set at center {center}:', val_metrics, file=log_file)

            # plot the integrated latents 
            plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in train_datasets],
                        zs_train[center].cpu().numpy(), output_dir, f'train-at-{center}-')
            plot_latent([d.dataset.dataset.get_adata()[d.dataset.indices] for d in val_datasets],
                        zs_val[center].cpu().numpy(), output_dir, f'val-at-{center}-')

        log_file.close()


def calc_metrics(adatas, zs):
    # concatenate all modal latents into one `AnnData`
    latent_all_adata = np.concatenate(zs, axis=0)
    latent_all_adata = sc.AnnData(latent_all_adata)
    obss = []
    for i, adata in enumerate(adatas):
        obs = adata.obs.copy(deep=True)
        obs['modal'] = f'modal-{i}'
        obss.append(obs)
    obss = pd.concat(obss)
    latent_all_adata.obs = obss

    ebm = entropy_batch_mixing(latent_all_adata, label_key='modal')
    knnp = knn_purity(latent_all_adata, label_key='modal')
    return {
        'EBM': ebm,
        'kNN-Purity': knnp
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('--root-dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(os.path.join(args.root_dir, 'config.json'))
    experiment_name = os.path.basename(args.root_dir)

    validate(experiment_name, **config)