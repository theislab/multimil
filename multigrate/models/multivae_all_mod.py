import torch
import scanpy as sc
import numpy as np
from .multivae_smaller import MultiVAE_smaller, MultiVAETorch_smaller
from .mlp_decoder import MLP_decoder
from .losses import MMD
from itertools import zip_longest

class MultiVAETorch_all_mod(MultiVAETorch_smaller):
    # TODO: matrix form? store in a dict
    def prep_latent(self, zs, pair_groups):
        # TODO: assumption: all have all modalities
        zs_corrected = []
        for z, pg in zip(zs, pair_groups):
            group_size = len(self.modalities_per_group[pg])
            mask = -torch.ones(group_size, self.n_modality)
            for i, mod in enumerate(self.modalities_per_group[pg]):
                mask[:, mod] += torch.eye(group_size)[i]
            mod_vecs = mask @ self.modality_vectors.weight
            mod_vecs = [mod_vecs[i] for i, _ in enumerate(mod_vecs)]
            z_group = [z + mod_vec for mod_vec in mod_vecs]
            zs_corrected.extend(z_group)

        return zs_corrected

class MultiVAE_all_mod(MultiVAE_smaller):
    def __init__(
        self,
        adatas,
        names,
        pair_groups,
        condition=None,
        normalization='layer',
        z_dim=15,
        h_dim=32,
        hiddens=[],
        losses=[],
        output_activations=[],
        shared_hiddens=[],
        recon_coef=1,
        kl_coef=1e-4,
        integ_coef=1e-1,
        cycle_coef=1e-2,
        dropout=0.2,
        device=None,
        loss_coefs=[],
        layers=[],
        theta=None
    ):
        super().__init__(adatas, names, pair_groups, condition,
        normalization,
        z_dim,
        h_dim,
        hiddens,
        losses,
        output_activations,
        shared_hiddens,
        recon_coef,
        kl_coef,
        integ_coef,
        cycle_coef,
        dropout,
        device,
        loss_coefs,
        layers,
        theta
        )

        self.model = MultiVAETorch_all_mod(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.device, self.condition, self.n_batch_labels,
                                   self.pair_groups_dict, self.modalities_per_group, self.paired_networks_per_modality_pairs)

    def impute_batch(self, x, pair, mod, batch, target_pair, target_modality):
        h = self.model.to_shared_dim(x, mod, batch)
        hs_concat, new_pair_group = self.model.encode_pairs([h], [pair])
        # fix
        z = self.model.bottleneck(hs_concat[0])
        mus = z[1]
        logvars = z[2]
        z = z[0]

        mask = -torch.ones(1, self.n_modality)
        mask[:, target_modality] += torch.ones(1)
        mod_vec = mask @ self.model.modality_vectors.weight
        z = z + mod_vec

        # fix batch label, so it takes mean of available batches
        r = self.model.decode_from_shared(z, target_modality, target_pair, 0)
        return r
