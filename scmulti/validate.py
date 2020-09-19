import argparse
import numpy as np
import pandas as pd
import time
import os
import json
from utils import create_optimizer, parse_config_file
from metrics import entropy_batch_mixing, knn_purity
from datasets import load_dataset
from models import load_model
from collections import OrderedDict
import matplotlib.pyplot as plt
import scanpy as sc
import torch


def validate(experiment_name, use_best_model=True, **config):
    # config torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['train']['seed'])

    # load train and validation datasets
    train_dataloader, val_dataloader = load_dataset(config['dataset'], device)
    
    # get configs
    output_dir = os.path.join(config['train']['output-dir'], experiment_name)
    log_file = open(os.path.join(output_dir, 'evaluations.txt'), 'a')

    # create the model to be trained
    model_prefix = 'best' if use_best_model else 'last'
    model_path = os.path.join(output_dir, f'{model_prefix}-model.pt')
    model = load_model(config['model'], model_path, device)

    # calculate validation loss
    # find integration of data
    with torch.no_grad():
        model.eval()

        # validate on training data
        train_loss = 0
        for xs, pair_indices in train_dataloader:
            loss, losses = model.test(xs, pair_indices)
            train_loss += loss.item()

        # validate on validation data
        val_loss = 0
        for xs, pair_indices in val_dataloader:
            loss, losses = model.test(xs, pair_indices)
            val_loss += loss.item()

        print(f'########## {model_prefix} model validation:', file=log_file)
        print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f}', file=log_file)

        # bring training datasets into latent space without integration
        zs_train_uninteg = OrderedDict()
        for xs, pair_indices in train_dataloader:
            for i, (name, x) in enumerate(zip(train_dataloader.dataset.names, xs)):
                z = model.to_latent(x, i)
                if name not in zs_train_uninteg:
                    zs_train_uninteg[name] = []
                zs_train_uninteg[name].append(z)
        for name, z in zs_train_uninteg.items():
            zs_train_uninteg[name] = torch.cat(z).cpu().numpy()

        # bring validation datasets into latent space without integration
        zs_val_uninteg = OrderedDict()
        for xs, pair_indices in val_dataloader:
            for i, (name, x) in enumerate(zip(val_dataloader.dataset.names, xs)):
                z = model.to_latent(x, i)
                if name not in zs_val_uninteg:
                    zs_val_uninteg[name] = []
                zs_val_uninteg[name].append(z)
        for name, z in zs_val_uninteg.items():
            zs_val_uninteg[name] = torch.cat(z).cpu().numpy()

        # calculate metrics for unintegrated latents
        train_metrics_uninteg = calc_metrics(train_dataloader.dataset.adatas, zs_train_uninteg)
        val_metrics_uninteg = calc_metrics(val_dataloader.dataset.adatas, zs_val_uninteg)
        train_metrics_uninteg = ', '.join([f'{k}={v:.4f}' for k, v in train_metrics_uninteg.items()])
        val_metrics_uninteg = ', '.join([f'{k}={v:.4f}' for k, v in val_metrics_uninteg.items()])
        print('metrics for unintegrated training set:', train_metrics_uninteg, file=log_file)
        print('metrics for unintegrated validation set:', val_metrics_uninteg, file=log_file)

        # plot the unintegrated latents 
        plot_latent(train_dataloader.dataset.adatas, zs_train_uninteg, output_dir, f'{model_prefix}-train-uintegrated-')
        plot_latent(val_dataloader.dataset.adatas, zs_val_uninteg, output_dir, f'{model_prefix}-val-unintegrated-')

        ### INTEGRATION ###
        # centers at which datasets should gather
        centers = {name: i for i, name in enumerate(train_dataloader.dataset.names)}
        centers['none'] = None

        # integrate training datasets into common latent space
        zs_train = {}
        for center_name, center_idx in centers.items():  # integrate datasets into multiple centers
            zs_train[center_name] = OrderedDict()
            for xs, pair_indices in train_dataloader:
                for i, (name, x) in enumerate(zip(train_dataloader.dataset.names, xs)):
                    z = model.integrate(x, i, center_idx)
                    if name not in zs_train[center_name]:
                        zs_train[center_name][name] = []
                    zs_train[center_name][name].append(z)
            for name, z in zs_train[center_name].items():
                zs_train[center_name][name] = torch.cat(z).cpu().numpy()

        # integrate validation datasets into common latent space
        zs_val = {}
        for cname, cidx in centers.items():  # integrate datasets into multiple centers
            zs_val[cname] = OrderedDict()
            for xs, pair_indices in val_dataloader:
                for i, (name, x) in enumerate(zip(val_dataloader.dataset.names, xs)):
                    z = model.integrate(x, i, cidx)
                    if name not in zs_val[cname]:
                        zs_val[cname][name] = []
                    zs_val[cname][name].append(z)
            for name, z in zs_val[cname].items():
                zs_val[cname][name] = torch.cat(z).cpu().numpy()

        # evaluate integration
        for cname in centers:
            # calculate metrics for integrated latents
            train_metrics = calc_metrics(train_dataloader.dataset.adatas, zs_train[cname])
            val_metrics = calc_metrics(val_dataloader.dataset.adatas, zs_val[cname])
            train_metrics = ', '.join([f'{k}={v:.4f}' for k, v in train_metrics.items()])
            val_metrics = ', '.join([f'{k}={v:.4f}' for k, v in val_metrics.items()])
            print(f'metrics for integrated training set at center {cname}:', train_metrics, file=log_file)
            print(f'metrics for integrated validation set at center {cname}:', val_metrics, file=log_file)

            # plot the integrated latents 
            plot_latent(train_dataloader.dataset.adatas, zs_train[cname], output_dir, f'{model_prefix}-train-at-{cname}-')
            plot_latent(val_dataloader.dataset.adatas, zs_val[cname], output_dir, f'{model_prefix}-val-at-{cname}-')

        log_file.close()


def calc_metrics(adatas, zs):
    # concatenate all modal latents into one `AnnData`
    latent_all_adata = np.concatenate(list(zs.values()), axis=0)
    latent_all_adata = sc.AnnData(latent_all_adata)
    obss = []
    for name, adata in adatas.items():
        obs = adata.obs.copy(deep=True)
        obs['modal'] = name
        obss.append(obs)
    obss = pd.concat(obss)
    latent_all_adata.obs = obss

    modal_ebm = entropy_batch_mixing(latent_all_adata, label_key='modal', n_neighbors=15, n_pools=15)
    modal_knnp = knn_purity(latent_all_adata, label_key='modal', n_neighbors=15)
    celltype_ebm = entropy_batch_mixing(latent_all_adata, label_key='cell_type', n_neighbors=15, n_pools=15)
    celltype_knnp = knn_purity(latent_all_adata, label_key='cell_type', n_neighbors=15)

    return {
        'Modal-EBM': modal_ebm,
        'Modal-kNNPurity': modal_knnp,
        'CellType-EBM': celltype_ebm,
        'CellType-kNNPurity': celltype_knnp
    }


def plot_latent(adatas, zs, save_dir, prefix='val-'):
    # plot a UMAP for each dataset
    for name, adata in adatas.items():
        latent_adata = sc.AnnData(zs[name])
        latent_adata.obs = adata.obs.copy(deep=True)
        sc.pp.neighbors(latent_adata)
        sc.tl.umap(latent_adata)
        sc.pl.umap(latent_adata, color=['cell_type'])
        plt.savefig(os.path.join(save_dir, f'{prefix}umap-latent-{name}.png'), dpi=200, bbox_inches='tight')

    # plot a UMAP for integrated datasets
    latent_all_adata = np.concatenate(list(zs.values()), axis=0)
    latent_all_adata = sc.AnnData(latent_all_adata)
    obss = []
    for name, adata in adatas.items():
        obs = adata.obs.copy(deep=True)
        obs['modal'] = name
        obss.append(obs)
    obss = pd.concat(obss)
    latent_all_adata.obs = obss

    sc.pp.neighbors(latent_all_adata)
    sc.tl.umap(latent_all_adata)
    sc.pl.umap(latent_all_adata, color=['modal', 'cell_type'], ncols=1)
    plt.savefig(os.path.join(save_dir, f'{prefix}umap-integrate-latents.png'), dpi=200, bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('--root-dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(os.path.join(args.root_dir, 'config.json'))
    experiment_name = os.path.basename(args.root_dir)

    validate(experiment_name, use_best_model=True, **config)
    validate(experiment_name, use_best_model=False, **config)