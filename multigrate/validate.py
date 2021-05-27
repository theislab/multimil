import argparse
import numpy as np
import pandas as pd
import time
import os
import json
from collections import OrderedDict
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import scanpy as sc
from .utils import parse_config_file, split_adatas
from .metrics import metrics
from .datasets import load_dataset
from .models import create_model
import torch

def validate(experiment_name, output_dir, config, save_losses=True):
    # load experiment configurations
    experiment_config = config['experiment']
    random_seed = experiment_config['seed']
    pair_split = experiment_config.get('pair-split', None)
    calc_matrics = experiment_config.get('calculate_metrics', True)
    impute = experiment_config.get('impute', True)
    batches_present = experiment_config['batch']
    batch_key = experiment_config.get('batch_key', 'batch')

    train_params = config['model']['train']
    if impute:
        impute_params = config['model']['impute']
    modality_key = train_params.get('modality_key', 'modality')
    celltype_key = train_params.get('celltype_key', 'cell_type')
    # torch.manual_seed(random_seed)

    # load adatas
    model_params = config['model']['params']
    adatas = []
    for adata_set in model_params['adatas']:
        adatas.append([])
        for adata_path in adata_set:
            adata = sc.read_h5ad(adata_path)
            adatas[-1].append(adata)
    model_params['adatas'] = adatas

    # recover the paired/unpaired splitting done by the training script
    if pair_split is not None:
        pair_group_masks = torch.load(os.path.join(output_dir, 'pair-group-masks.pt'))
        adatas, names, pair_groups, _ = split_adatas(adatas, model_params['names'], model_params['pair_groups'], pair_split, pair_group_masks, shuffle_unpaired=False)
        model_params['adatas'] = adatas
        model_params['names'] = names
        model_params['pair_groups'] = pair_groups

    # load the model
    model = create_model(config['model']['name'], model_params)
    model.load(output_dir)
    if save_losses:
        save_losses_figure(model, output_dir)

    # validate the model
    with torch.no_grad():
        # predict the shared latent space
        out = model.test(
            adatas,
            names,
            pair_groups=pair_groups,
            layers=model_params['layers'],
            batch_size=train_params['batch_size']
        )

        # TODO check if layers are working
        for adata in out:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)

            adata.obs['batch'] = (adata.obs['study'] + '-modality-' + adata.obs['modality'].astype('str')).astype('category')
            adata.obs['modality'] = adata.obs['modality'].astype('category')

        integrated, latent, corrected, hs = out

        sc.pl.umap(hs, color=['batch', 'study', 'modality', 'cell_type'], ncols=2, wspace=0.4, show=False)
        plt.savefig(os.path.join(output_dir, 'hs.png'), dpi=200, bbox_inches='tight')

        sc.pl.umap(corrected, color=['batch', 'study', 'modality', 'cell_type'], ncols=2, wspace=0.4, show=False)
        plt.savefig(os.path.join(output_dir, 'corrected.png'), dpi=200, bbox_inches='tight')

        sc.pl.umap(latent, color=['batch', 'study', 'modality', 'cell_type'], ncols=2, wspace=0.4, show=False)
        plt.savefig(os.path.join(output_dir, 'latent.png'), dpi=200, bbox_inches='tight')

        sc.pl.umap(integrated, color=['study', 'modality', 'cell_type'], ncols=2, wspace=0.4, show=False)
        plt.savefig(os.path.join(output_dir, 'integrated.png'), dpi=200, bbox_inches='tight')

        plt.close('all')


        if calc_matrics:
            integrated.obsm['X_latent'] = integrated.X
            mtrcs = metrics(
                None, integrated,
                batch_key=batch_key,
                label_key=celltype_key,
                embed='X_latent',
                pcr_batch=False,
                isolated_label_f1=False,
                asw_batch=batches_present
            )
            print(mtrcs.to_dict())
            json.dump(mtrcs.to_dict()['score'], open(os.path.join(output_dir, 'metrics.json'), 'w'), indent=2)

        # impute
        if impute:
            adatas = []
            for adata_set in impute_params['adatas']:
                adatas.append([])
                for adata_path in adata_set:
                    adata = sc.read_h5ad(adata_path)
                    adatas[-1].append(adata)

            true_protein = sc.read_h5ad(impute_params['true_protein_adata'])

            r = model.impute(
                adatas = adatas,
                names = impute_params['names'],
                pair_groups = impute_params['pair_groups'],
                target_modality = impute_params['target_modality'],
                batch_labels = impute_params['batch_labels'],
                target_pair = impute_params['target_pair'],
                layers=impute_params['layers'],
                batch_size=train_params['batch_size'],
            )
            r.obsm['predicted_protein'] = r.X

            true_protein.obsm['true_protein'] = true_protein.X

            protein_corrs = pd.DataFrame()
            for i, protein in enumerate(true_protein.var_names):
                value = np.round(pearsonr(r.obsm['predicted_protein'][:, i], true_protein.obsm['true_protein'][:, i])[0], 3)
                protein_corrs = protein_corrs.append({'protein': protein, 'correlation': value}, ignore_index=True)

            protein_corrs = protein_corrs.append({'protein': 'mean', 'correlation': protein_corrs['correlation'].mean().round(3)}, ignore_index=True)
            protein_corrs = protein_corrs.set_index('protein')
            protein_corrs.to_csv(os.path.join(output_dir, 'protein_correlation.csv'))

def parse_args():
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('--root-dir', type=str)
    return parser.parse_args()

def save_losses_figure(model, output_dir):
    plt.figure(figsize=(15, 20));
    loss_names = ['recon', 'kl', 'integ']#, 'recon_mse', 'recon_nb']#, 'recon_bce']
    # nrows = int(np.ceil((len(loss_names)+1)/2))
    nrows = 3

    plt.subplot(nrows, 2, 1)
    plt.plot(model.history['iteration'], model.history['train_loss'], '.-', label='Train loss');
    plt.plot(model.history['iteration'], model.history['val_loss'], '.-', label='Val loss');
    plt.xlabel('#Iterations');
    plt.legend()

    for i, name in enumerate(loss_names):
        plt.subplot(nrows, 2, i+2)
        plt.plot(model.history['iteration'], model.history[f'train_{name}'], '.-', label=f'Train {name} loss');
        plt.plot(model.history['iteration'], model.history[f'val_{name}'], '.-', label=f'Val {name} loss');
        plt.xlabel('#Iterations');
        plt.legend()

    #plt.subplot(nrows, 2, nrows*2)
    #plt.plot(model.history['iteration'], model.history['mod_vec0_norm'], '.-', label='mod vec 1 norm');
    #plt.plot(model.history['iteration'], model.history['mod_vec1_norm'], '.-', label='mod vec 2 norm');
    #plt.plot(model.history['iteration'], model.history['mod_vec2_norm'], '.-', label='mod vec 3 norm');
    #plt.xlabel('#Iterations');
    #plt.legend()

    plt.savefig(os.path.join(output_dir, f'losses.png'), dpi=80, bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(os.path.join(args.root_dir, 'config.json'))
    experiment_name = os.path.basename(args.root_dir)

    validate(experiment_name, args.root_dir, config)
