import argparse
import numpy as np
import pandas as pd
import os
import json
from matplotlib import pyplot as plt
from utils import parse_config_file


def analyze(root_dir):
    # read experiments
    dirs = os.listdir(root_dir)
    experiments = [] 
    for dir in dirs:
        try:
            experiment = read_experiment(os.path.join(root_dir, dir))
            experiments.append(experiment)
        except:
            pass
    experiments = pd.DataFrame(experiments)

    analyze_dir = os.path.join(root_dir, 'analyze')
    os.makedirs(analyze_dir, exist_ok=True)

    # plot the effect of different KL weights 
    unique_settings = experiments[['recon-coef', 'integ-coef', 'cycle-coef', 'adversarial', 'pair-split']].drop_duplicates()
    for _, setting in unique_settings.iterrows():
        exp = experiments[(experiments['recon-coef'] == setting['recon-coef']) & \
                          (experiments['integ-coef'] == setting['integ-coef']) & \
                          (experiments['cycle-coef'] == setting['cycle-coef']) & \
                          (experiments['adversarial'] == setting['adversarial']) & \
                          (experiments['pair-split'] == setting['pair-split'])]
        exp_name = f'recon{setting["recon-coef"]}_' + \
                   f'integ{setting["integ-coef"]}_' + \
                   f'cycle{setting["cycle-coef"]}_' + \
                   f'adver{int(setting["adversarial"])}_' + \
                   f'pair{setting["pair-split"]}'
        plot_kl_coef_effect(exp, [
            'ASW_label/batch',
            'graph_conn',
            'ASW_label',
            # 'ARI_cluster/label',
            'NMI_cluster/label',
            'isolated_label_silhouette'
        ], analyze_dir, prefix=f'kl-{exp_name}')

    # plot semi-supervised settings: the effect of pairness
    unique_settings = experiments[['recon-coef', 'kl-coef', 'integ-coef', 'cycle-coef', 'adversarial']].drop_duplicates()
    for _, setting in unique_settings.iterrows():
        exp = experiments[(experiments['recon-coef'] == setting['recon-coef']) & \
                          (experiments['kl-coef'] == setting['kl-coef']) & \
                          (experiments['integ-coef'] == setting['integ-coef']) & \
                          (experiments['cycle-coef'] == setting['cycle-coef']) & \
                          (experiments['adversarial'] == setting['adversarial'])]
        exp_name = exp.iloc[0]['name']
        plot_semisupervised(exp, [
            'ASW_label/batch',
            'graph_conn',
            'ASW_label',
            # 'ARI_cluster/label',
            'NMI_cluster/label',
            'isolated_label_silhouette'
        ], analyze_dir, prefix=f'semisupervised-{exp_name}')
        
    # plot the effect of integration loss
    unique_settings = experiments[['recon-coef', 'kl-coef', 'cycle-coef', 'adversarial', 'pair-split']].drop_duplicates()
    for _, setting in unique_settings.iterrows():
        exp = experiments[(experiments['recon-coef'] == setting['recon-coef']) & \
                          (experiments['kl-coef'] == setting['kl-coef']) & \
                          (experiments['cycle-coef'] == setting['cycle-coef']) & \
                          (experiments['adversarial'] == setting['adversarial']) & \
                          (experiments['pair-split'] == setting['pair-split'])]
        exp_name = f'recon{setting["recon-coef"]}_' + \
                   f'kl{setting["kl-coef"]}_' + \
                   f'cycle{setting["cycle-coef"]}_' + \
                   f'adver{int(setting["adversarial"])}_' + \
                   f'pair{setting["pair-split"]}'
        plot_integ_coef_effect(exp, [
            'ASW_label/batch',
            'graph_conn',
            'ASW_label',
            # 'ARI_cluster/label',
            'NMI_cluster/label',
            'isolated_label_silhouette'
        ], analyze_dir, prefix=f'integration-{exp_name}')
    
    # plot the effect of cycle consistency loss
    unique_settings = experiments[['recon-coef', 'kl-coef', 'integ-coef', 'adversarial', 'pair-split']].drop_duplicates()
    for _, setting in unique_settings.iterrows():
        exp = experiments[(experiments['recon-coef'] == setting['recon-coef']) & \
                          (experiments['kl-coef'] == setting['kl-coef']) & \
                          (experiments['integ-coef'] == setting['integ-coef']) & \
                          (experiments['adversarial'] == setting['adversarial']) & \
                          (experiments['pair-split'] == setting['pair-split'])]
        exp_name = f'recon{setting["recon-coef"]}_' + \
                   f'kl{setting["kl-coef"]}_' + \
                   f'integ{setting["integ-coef"]}_' + \
                   f'adver{int(setting["adversarial"])}_' + \
                   f'pair{setting["pair-split"]}'
        plot_cycle_consistency(exp, [
            'ASW_label/batch',
            'graph_conn',
            'ASW_label',
            # 'ARI_cluster/label',
            'NMI_cluster/label',
            'isolated_label_silhouette'
        ], analyze_dir, prefix=f'cycleconsistency-{exp_name}')

    
def read_experiment(dir):
    config = parse_config_file(os.path.join(dir, 'config.json'))

    model_params = config['model']['params']
    experiment_config = config['experiment']

    exp_name = '_'.join(dir.split('_')[1:-1])
    experiment = {'name': exp_name, 'dir': dir}
    for key in ['recon_coef', 'kl_coef', 'integ_coef', 'cycle_coef', 'adversarial']:
        experiment[key.replace('_', '-')] = model_params[key]
    for key in ['pair-split', 'seed']:
        experiment[key] = experiment_config[key]

    last_model_metrics = parse_config_file(os.path.join(dir, 'metrics.json'))
    experiment.update(last_model_metrics)

    return experiment


def flatten_dict(d, prefix=''):
    # TODO remove
    flat_d = {} 
    for key in d:
        key_prefix = key if prefix == '' else f'{prefix}-{key}'
        if type(d[key]) == dict:
            flat_dkey = flatten_dict(d[key])
            flat_dkey = {f'{key_prefix}-{k}': v for k, v in flat_dkey.items()}
            flat_d.update(flat_dkey)
        else:
            flat_d[key_prefix] = d[key]
    return flat_d


def plot_semisupervised(experiments, metrics, save_dir, prefix=''):
    experiments = experiments.sort_values('pair-split')

    # plot metrics
    for metric in metrics:
        plt.plot(experiments['pair-split'], experiments[metric], '.-', label=metric)
    plt.legend()
    plt.xlabel('Paired data')
    plt.ylabel('Metrics')
    plt.savefig(os.path.join(save_dir, f'{prefix}.png'), dpi=200, bbox_inches='tight')
    plt.clf()

    # plot umaps
    umaps = []
    for _, exp in experiments.iterrows():
        umap = plt.imread(os.path.join(exp['dir'], 'umap-z.png'))
        umaps.append(umap)
    umap_all = np.concatenate(umaps, axis=1)
    plt.imsave(os.path.join(save_dir, f'{prefix}-umaps.png'), umap_all, dpi=200)
    plt.clf()


def plot_kl_coef_effect(experiments, metrics, save_dir, prefix=''):
    experiments = experiments.sort_values('kl-coef')
    for metric in metrics:
        plt.plot(np.log10(experiments['kl-coef']), experiments[metric], '.-', label=metric)
    plt.legend()
    plt.xlabel('KL weight (log10 scale)')
    plt.ylabel('Metrics')
    plt.savefig(os.path.join(save_dir, f'{prefix}.png'), dpi=200, bbox_inches='tight')
    plt.clf()

    # plot umaps
    umaps = []
    for _, exp in experiments.iterrows():
        umap = plt.imread(os.path.join(exp['dir'], 'umap-z.png'))
        umaps.append(umap)
    umap_all = np.concatenate(umaps, axis=1)
    plt.imsave(os.path.join(save_dir, f'{prefix}-umaps.png'), umap_all, dpi=200)
    plt.clf()


def plot_integ_coef_effect(experiments, metrics, save_dir, prefix=''):
    experiments = experiments.sort_values('integ-coef')
    experiments = experiments[experiments['integ-coef'] > 0]  # we want integration!
    for metric in metrics:
        plt.plot(np.log10(experiments['integ-coef']), experiments[metric], '.-', label=metric)
    plt.legend()
    plt.xlabel('Integration weight (log scale)')
    plt.ylabel('Metrics')
    plt.savefig(os.path.join(save_dir, f'{prefix}.png'), dpi=200, bbox_inches='tight')
    plt.clf()

    # plot umaps
    umaps = []
    for _, exp in experiments.iterrows():
        umap = plt.imread(os.path.join(exp['dir'], 'umap-z.png'))
        umaps.append(umap)
    umap_all = np.concatenate(umaps, axis=1)
    plt.imsave(os.path.join(save_dir, f'{prefix}-umaps.png'), umap_all, dpi=200)
    plt.clf()


def plot_cycle_consistency(experiments, metrics, save_dir, prefix=''):
    experiments = experiments.sort_values('cycle-coef')
    experiments = experiments[experiments['cycle-coef'] > 0]  # we want cycle consistency!
    for metric in metrics:
        plt.plot(np.log10(experiments['cycle-coef']), experiments[metric], '.-', label=metric)
    plt.legend()
    plt.xlabel('Cycle consistency weight (log scale)')
    plt.ylabel('Metrics')
    plt.savefig(os.path.join(save_dir, f'{prefix}.png'), dpi=200, bbox_inches='tight')
    plt.clf()

    # plot umaps
    umaps = []
    for _, exp in experiments.iterrows():
        umap = plt.imread(os.path.join(exp['dir'], 'umap-z.png'))
        umaps.append(umap)
    if umaps:
        umap_all = np.concatenate(umaps, axis=1)
        plt.imsave(os.path.join(save_dir, f'{prefix}-umaps.png'), umap_all, dpi=200)
        plt.clf()
            

def parse_args():
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--root-dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    analyze(args.root_dir)