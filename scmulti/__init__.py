import copy
from collections import defaultdict

from . import utils
from . import models
from . import datasets
from . import metrics

from .models.multivae import MultiVAETorch
from .models.mlp import MLP

__author__ = ', '.join([
    'Alireza Omidi',
    'Anastasia Litinetskaya',
    'Mohammad Lotfollahi',
])

__email__ = ', '.join([
    'ga58som@mytum.de', #test
])

def operate(network, adatas, names, pair_groups, batch_labels):

    if len(adatas) != network.model.n_modality:
        raise ValueError(f'new modalities and the old modalities must be the same length. new_modalities = {len(adatas)} != {network.model.n_modality} = old_modalities')

    new_network = copy.deepcopy(network)

    # TODO: figure out why doesn't work
    #new_network.reset_history()
    new_network._train_history = defaultdict(list)
    new_network._val_history = defaultdict(list)

    new_network.adatas = new_network.reshape_adatas(adatas, names, pair_groups, batch_labels)

    old_encoders = network.encoders
    old_decoders = network.decoders

    old_shared_encoder = network.shared_encoder
    old_shared_decoder = network.shared_decoder

    # new batch labels to add
    new_n_batch_labels = len(set([item for sublist in batch_labels for item in sublist]) | set([item for sublist in network.batch_labels for item in sublist]))
    new_n_batch_and_mod_labels = network.n_modality + new_n_batch_labels

    # cVAE version 1
    if network.condition == '0':
        encoders = [MLP(x_dim + new_n_batch_labels, network.h_dim, hs, output_activation='leakyrelu',
                                 dropout=network.dropout, batch_norm=True, regularize_last_layer=True) for x_dim, hs in zip(network.x_dims, network.hiddens)]
        shared_decoder = MLP(network.z_dim + new_n_batch_labels, network.h_dim, network.shared_hiddens[::-1], output_activation='leakyrelu',
                                      dropout=network.dropout, batch_norm=True, regularize_last_layer=True)

        new_network.model = MultiVAETorch(encoders, network.decoders, network.shared_encoder, shared_decoder,
                                       network.mu, network.logvar, network.modality_vecs, network.adv_disc, network.device, network.condition, new_n_batch_labels)

    # cVAE version 2
    elif network.condition == '1':
        decoders = [MLP(network.h_dim + new_n_batch_and_mod_labels, x_dim, hs[::-1], output_activation=out_act,
                             dropout=network.dropout, batch_norm=True) for x_dim, hs, out_act in zip(network.x_dims, network.hiddens, network.output_activations)]
        shared_encoder = MLP(network.h_dim + new_n_batch_and_mod_labels, network.z_dim, network.shared_hiddens, output_activation='leakyrelu',
                                  dropout=network.dropout, batch_norm=True, regularize_last_layer=True)
        new_network.model = MultiVAETorch(network.encoders, decoders, shared_encoder, network.shared_decoder,
                                   network.mu, network.logvar, network.modality_vecs, network.adv_disc, network.device, network.condition, new_n_batch_labels)
    else:
        raise NotImplementedError(f'Condition is not implemented. Should be either "0", "1" or None.')

    # set new weights to new weights when possible and freeze some weights 
    for (old_p, new_p) in zip(network.model.named_parameters(), new_network.model.named_parameters()):
        if old_p[0] == new_p[0]:
            name = old_p[0]
            if old_p[1].shape == new_p[1].shape:
                new_network.model.state_dict()[name] = network.model.state_dict()[name]
                new_p[1].requires_grad = False
            else:
                old_len = network.model.state_dict()[name].shape[1]
                new_network.model.state_dict()[name][:, :old_len] = network.model.state_dict()[name]

    return new_network
