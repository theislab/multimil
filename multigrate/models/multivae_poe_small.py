import torch
from itertools import groupby
from .multivae_poe import MultiVAE_PoE, MultiVAETorch_PoE

class MultiVAETorch_PoE_small(MultiVAETorch_PoE):

    def forward(self, xs, modalities, pair_groups, batch_labels, size_factors):
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        zs = [self.bottleneck(h) for h in hs]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        mus_joint, logvars_joint, _ = self.product_of_experts(mus, logvars, pair_groups)
        zs_joint = [self.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
        out = self.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels)
        zs = out[0]
        rs = [self.decode_from_shared(z, mod, pair_group, batch_label) for z, mod, pair_group, batch_label in zip(zs, modalities, pair_groups, batch_labels)]
        return rs, zs, mus+mus_joint, logvars+logvars_joint, pair_groups, modalities, batch_labels, xs, size_factors

    def prep_latent(self, xs, zs, zs_joint, modalities, pair_groups, batch_labels, size_factors=None):
        zs_new = []
        current_joint = 0 # index for joint zs
        current = 0
        for pair, group in groupby(pair_groups):
            group_size = len(list(group))
            if group_size == 1:
                zs_new.append(zs[current])
                current += 1
                continue

            mask = torch.zeros(group_size, self.n_modality)
            for i, mod in enumerate(self.modalities_per_group[pair]):
                mask[:, mod] = -torch.ones(group_size)
                mask[:, mod] += torch.eye(group_size)[i]
            mod_vecs = mask @ self.modality_vectors.weight
            mod_vecs = [mod_vecs[i] for i, _ in enumerate(mod_vecs)]
            z_group = [zs_joint[current_joint] + mod_vec for mod_vec in mod_vecs]
            zs_new.extend(z_group)

            current_joint += 1
            current += group_size
        return zs_new, modalities, pair_groups, batch_labels, xs, size_factors

    def convert(self, z, target_modality):
        return z + self.modal_vector(target_modality) @ self.modality_vectors.weight

class MultiVAE_PoE_small(MultiVAE_PoE):
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

        self.model = MultiVAETorch_PoE_small(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.device, self.condition, self.n_batch_labels,
                                   self.pair_groups_dict, self.modalities_per_group, self.paired_networks_per_modality_pairs)

    def impute_batch(self, x, pair, mod, batch, target_pair, target_modality):
        h = self.model.to_shared_dim(x, mod, batch)
        z = self.model.bottleneck(h)
        mus = z[1]
        logvars = z[2]
        z = z[0]
        z = self.model.convert(z, target_modality)
        # TODO fix batches
        r = self.model.decode_from_shared(z, target_modality, pair, 0)
        return r
