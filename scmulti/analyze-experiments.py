import argparse
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

    # plot modal-asw vs celltype-asw for paired setting
    # experiments_paired = experiments[experiments['pair-split'] == 1]
    # if not experiments_paired.empty:
    #     plot_vs_metrics(experiments_paired, 'best-val-none-modal-asw', 'best-val-none-cell_type-asw', analyze_dir, prefix='paired-')

    # plot modal-asw vs celltype-asw for unpaired setting
    # experiments_unpaired = experiments[experiments['pair-split'] == 0]
    # if not experiments_unpaired.empty:
    #     plot_vs_metrics(experiments_unpaired, 'best-val-none-modal-asw', 'best-val-none-cell_type-asw', analyze_dir, prefix='unpaired-')
    
    # plot semi-supervised settings
    unique_settings = experiments[['recon-coef', 'kl-coef', 'integ-coef', 'cross-coef', 'cycle-coef', 'adversarial']].drop_duplicates()
    for _, setting in unique_settings.iterrows():
        exp = experiments[(experiments['recon-coef'] == setting['recon-coef']) & \
                          (experiments['kl-coef'] == setting['kl-coef']) & \
                          (experiments['integ-coef'] == setting['integ-coef']) & \
                          (experiments['cross-coef'] == setting['cross-coef']) & \
                          (experiments['cycle-coef'] == setting['cycle-coef']) & \
                          (experiments['adversarial'] == setting['adversarial'])]
        exp_name = exp.iloc[0]['name']
        for model_type in ['best', 'last']:
            plot_semisupervised(exp, [
                f'{model_type}-val-none-ASW_label/batch',
                f'{model_type}-val-none-ASW_label',
                f'{model_type}-val-none-ARI_cluster/label',
                f'{model_type}-val-none-NMI_cluster/label'
                
            ], analyze_dir, prefix=f'semisupervised-{exp_name}-{model_type}')

    
def read_experiment(dir):
    config = parse_config_file(os.path.join(dir, 'config.json'))
    best_model_metrics = parse_config_file(os.path.join(dir, 'best-model-evaluations.json')) 
    last_model_metrics = parse_config_file(os.path.join(dir, 'last-model-evaluations.json'))

    model_params = config['model']['params']
    dataset_params = config['dataset']

    exp_name = '_'.join(dir.split('_')[1:-1])
    experiment = {'name': exp_name}
    for key in ['recon_coef', 'kl_coef', 'integ_coef', 'cross_coef', 'cycle_coef', 'adversarial']:
        experiment[key.replace('_', '-')] = model_params[key]
    for key in ['pair-split', 'seed']:
        experiment[key] = dataset_params[key]

    best_model_metrics = flatten_dict(best_model_metrics, prefix='best')
    last_model_metrics = flatten_dict(last_model_metrics, prefix='last')
    experiment.update(best_model_metrics)
    experiment.update(last_model_metrics)

    return experiment


def flatten_dict(d, prefix=''):
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


def plot_vs_metrics(df, metric1, metric2, save_dir, prefix=''):
    plt.scatter(df[metric1], df[metric2])
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    plt.savefig(os.path.join(save_dir, f'{prefix}{metric1}_vs_{metric2}'), dpi=200, bbox_inches='tight')
    plt.clf()


def plot_semisupervised(experiment, metrics, save_dir, prefix=''):
    experiment = experiment.sort_values('pair-split')
    for metric in metrics:
        plt.plot(experiment['pair-split'], experiment[metric], '.-', label=metric)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{prefix}.png'), dpi=200, bbox_inches='tight')
    plt.clf()
            

def parse_args():
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--root-dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    analyze(args.root_dir)