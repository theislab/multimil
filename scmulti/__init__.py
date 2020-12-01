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

def operate(network, adatas, names, pair_groups):

    if len(adatas) != network.model.n_modality:
        raise ValueError(f'new modalities and the old modalities must be the same length. new_modalities = {len(adatas)} != {network.model.n_modality} = old_modalities')

    new_network = copy.deepcopy(network)

    # TODO: figure out why doesn't work
    #new_network.reset_history()
    new_network._train_history = defaultdict(list)
    new_network._val_history = defaultdict(list)

    n_new_batch_labels = [len(modality_adatas) for modality_adatas in adatas]
    new_network.n_batch_labels = [new + old for new, old in zip(n_new_batch_labels, network.n_batch_labels)]

    batch_labels = [list(range(len(modality_adatas))) for modality_adatas in adatas]
    #print(batch_labels)

    for i, modality_batch_labels in enumerate(batch_labels):
        for j, batch_label in enumerate(modality_batch_labels):
            modality_batch_labels[j] = batch_label + network.n_batch_labels[i]
    new_network.batch_labels = batch_labels
#    print('old')
    #print(network.n_batch_labels)
    #print(network.batch_labels)
    #print('new')
    #print(n_new_batch_labels)
    #print(batch_labels)

    new_network.adatas = new_network.reshape_adatas(adatas, names, pair_groups, batch_labels)

#    old_encoders = network.encoders
#    old_decoders = network.decoders

#    old_shared_encoder = network.shared_encoder
#    old_shared_decoder = network.shared_decoder

    #print(n_new_batch_labels)
    if new_network.condition:
        encoders = [MLP(x_dim + new_network.n_batch_labels[i], network.h_dim, hs, output_activation='leakyrelu',
                                 dropout=network.dropout, batch_norm=True, regularize_last_layer=True) for i, (x_dim, hs) in enumerate(zip(network.x_dims, network.hiddens))]
        decoders = [MLP(network.h_dim + new_network.n_batch_labels[i], x_dim, hs[::-1], output_activation=out_act,
                             dropout=network.dropout, batch_norm=True) for i, (x_dim, hs, out_act) in enumerate(zip(network.x_dims, network.hiddens, network.output_activations))]
        new_network.model = MultiVAETorch(encoders, decoders, network.shared_encoder, network.shared_decoder,
                                   network.mu, network.logvar, network.modality_vecs, network.adv_disc, network.device, network.condition, new_network.n_batch_labels)
    else:
        raise NotImplementedError(f'The original network should be conditioned (either "0" or "1").')

    # set new weights to old weights when possible and freeze some weights
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
