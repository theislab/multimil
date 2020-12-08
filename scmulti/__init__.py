import copy
import torch
from torch import nn

from . import utils
from . import models
from . import datasets
from . import metrics

from .models.multivae import MultiVAETorch, MultiVAE
from .models.mlp import MLP

__author__ = ', '.join([
    'Alireza Omidi',
    'Anastasia Litinetskaya',
    'Mohammad Lotfollahi',
])

__email__ = ', '.join([
    'ga58som@mytum.de', #test
])

def operate(network, adatas, names, pair_groups, fine_tune='cond_weights'):

    if len(adatas) != network.model.n_modality:
        raise ValueError(f'new modalities and the old modalities must be the same length. new_modalities = {len(adatas)} != {network.model.n_modality} = old_modalities')

    new_network = MultiVAE(adatas, names, pair_groups, condition=network.condition,
                                    z_dim=network.z_dim,
                                    h_dim=network.h_dim,
                                    hiddens=network.hiddens,
                                    output_activations=network.output_activations,
                                    shared_hiddens=network.shared_hiddens,
                                    adver_hiddens=network.adver_hiddens,
                                    recon_coef=network.recon_coef,
                                    kl_coef=network.kl_coef,
                                    integ_coef=network.integ_coef,
                                    cycle_coef=network.cycle_coef,
                                    adversarial=network.adversarial,
                                    dropout=network.dropout)

    n_new_batch_labels = [len(modality_adatas) for modality_adatas in adatas]
    changed_modalities_idx = []
    for i, n in enumerate(n_new_batch_labels):
        if n>0:
            changed_modalities_idx.append(i)

    new_network.n_batch_labels = [new + old for new, old in zip(n_new_batch_labels, network.n_batch_labels)]

    batch_labels = [list(range(len(modality_adatas))) for modality_adatas in adatas]
    batch_labels = [[batch_label + n_batch_labels[i] for j, batch_label in enumerate(modality_batch_labels)] for i, modality_batch_labels in enumerate(batch_labels)]

    new_network.batch_labels = batch_labels
    new_network.adatas = new_network.reshape_adatas(adatas, names, pair_groups, batch_labels)
    new_network.x_dims = network.x_dims

    if new_network.condition:
        encoders = [MLP(x_dim + new_network.n_batch_labels[i], network.h_dim, hs, output_activation='leakyrelu',
                                 dropout=network.dropout, batch_norm=True, regularize_last_layer=True) for i, (x_dim, hs) in enumerate(zip(network.x_dims, network.hiddens))]
        decoders = [MLP(network.h_dim + new_network.n_batch_labels[i], x_dim, hs[::-1], output_activation=out_act,
                             dropout=network.dropout, batch_norm=True) for i, (x_dim, hs, out_act) in enumerate(zip(network.x_dims, network.hiddens, network.output_activations))]
        new_network.model = MultiVAETorch(encoders, decoders, copy.deepcopy(network.shared_encoder), copy.deepcopy(network.shared_decoder),
                                   copy.deepcopy(network.mu), copy.deepcopy(network.logvar), copy.deepcopy(network.modality_vecs), copy.deepcopy(network.adv_disc), network.device, network.condition, new_network.n_batch_labels)
    else:
        raise NotImplementedError('The original network should be conditioned (either "0" or "1").')

    # set new weights to old weights when possible and freeze some weights
    for (old_p, new_p) in zip(network.model.named_parameters(), new_network.model.named_parameters()):
        if old_p[0] == new_p[0]:
            name = old_p[0]
            if old_p[1].shape == new_p[1].shape:
                new_p[1].data.copy_(old_p[1].data)
                new_p[1].requires_grad = False
            else:
                old_len = network.model.state_dict()[name].shape[1]
                new_network.model.state_dict()[name][:, :old_len] = network.model.state_dict()[name]

    if fine_tune == 'cond_weights':
        for name, layer in new_network.model.named_modules():
            if isinstance(layer, nn.Linear) and (name.startswith('encoder_') or name.startswith('decoder_')) and int(name[8]) in changed_modalities_idx:
                n_idx_to_freeze = n_new_batch_labels[int(name[8])]
                freeze_linear_params(layer, n_idx_to_freeze)

    return new_network

def freeze_linear_params(layer, n_idx_to_freeze):
    # adjusted from https://github.com/galidor/PyTorchPartialLayerFreezing/blob/main/partial_freezing.py
    # note that in our architecture bias is never fine-tuned
    def freezing_hook_weight_full(grad, weight_multiplier):
        return grad * weight_multiplier

    def freezing_hook_bias_full(grad, bias_multiplier):
        return grad * bias_multiplier

    weight_multiplier = torch.ones(layer.weight.shape[1]).to(layer.weight.device)
    weight_multiplier[:-n_idx_to_freeze] = 0
    weight_multiplier = weight_multiplier.view(1, -1)
    freezing_hook_weight = lambda grad: freezing_hook_weight_full(grad, weight_multiplier)

    weight_hook_handle = layer.weight.register_hook(freezing_hook_weight)

    return weight_hook_handle
