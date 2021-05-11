import torch
import scanpy as sc
import numpy as np
from .multivae_smaller import MultiVAE_smaller, MultiVAETorch_smaller
from .mlp import MLP
from itertools import groupby, zip_longest
from .losses import MMD

class MultiVAETorch_PoE(MultiVAETorch_smaller):

    def forward(self, xs, modalities, pair_groups, batch_labels, size_factors):
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        zs = [self.bottleneck(h) for h in hs]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        mus_joint, logvars_joint, _ = self.product_of_experts(mus, logvars, pair_groups)
        zs_joint = [self.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
        zs, modalities, pair_groups, batch_labels, xs, size_factors = self.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels, size_factors)
        rs = [self.decode_from_shared(z, mod, pair_group, batch_label) for z, mod, pair_group, batch_label in zip(zs, modalities, pair_groups, batch_labels)]
        return rs, zs, mus+mus_joint, logvars+logvars_joint, pair_groups, modalities, batch_labels, xs, size_factors

    def prep_latent(self, xs, zs, zs_joint, modalities, pair_groups, batch_labels, size_factors=None):
        zs_new, mods_new, pgs_new, bls_new, xs_new, sf_new = [], [], [], [], [], []
        current_joint = 0 # index for joint zs
        current = 0
        for pair, group in groupby(pair_groups):
            group_size = len(list(group))
            if group_size == 1:
                xs_new.append(xs[current])
                zs_new.append(zs[current])
                mods_new.append(modalities[current])
                pgs_new.append(pair_groups[current])
                bls_new.append(batch_labels[current])
                if size_factors:
                    sf_new.append(size_factors[current])
                current += 1
                continue
            xs_new.extend(xs[current:current+group_size]*group_size)
            if size_factors:
                sf_new.extend(size_factors[current:current+group_size]*group_size)
            zs_new.extend(zs[current:current+group_size])
            zs_new.extend([zs_joint[current_joint]]*group_size)
            mods_new.extend(modalities[current:current+group_size]*group_size)
            pgs_new.extend(pair_groups[current:current+group_size]*group_size)
            bls_new.extend(batch_labels[current:current+group_size]*group_size)
            current_joint += 1
            current += group_size
        return zs_new, mods_new, pgs_new, bls_new, xs_new, sf_new

    def product_of_experts(self, mus, logvars, pair_groups):
        # TODO cite
        joint_pair_groups = []
        mus_joint = []
        logvars_joint = []
        current = 0
        for pair, group in groupby(pair_groups):
            group_size = len(list(group))
            if group_size == 1:
                current += 1
                continue
		    # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
            logvar_joint = 1.0
            # formula: mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint, where mu_prior = 0.0
            mu_joint = 0.0

            for i in range(current, current+group_size):
                logvar_joint += 1.0 / torch.exp(logvars[i]) # sum up all inverse vars, logvars first needs to be converted to var, last 1.0 is coming from the prior
                mu_joint += mus[i] * (1.0 / torch.exp(logvars[i]))

            current += group_size
            logvar_joint = torch.log(1.0 / logvar_joint)  # inverse and convert to logvar
            mu_joint = mu_joint * torch.exp(logvar_joint)

            mus_joint.append(mu_joint)
            logvars_joint.append(logvar_joint)
            joint_pair_groups.append(pair)

        return mus_joint, logvars_joint, joint_pair_groups

class MultiVAE_PoE(MultiVAE_smaller):
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

        self.encoders = [MLP(x_dim + self.n_batch_labels[i], z_dim, hs, output_activation='leakyrelu',
                             dropout=dropout, norm=normalization, regularize_last_layer=True) if x_dim > 0 else None for i, (x_dim, hs) in enumerate(zip(self.x_dims, hiddens))]

        self.model = MultiVAETorch_PoE(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.device, self.condition, self.n_batch_labels,
                                   self.pair_groups_dict, self.modalities_per_group, self.paired_networks_per_modality_pairs)

    def test(self,
            adatas,
            names,
            pair_groups,
            modality_key='modality',
            celltype_key='cell_type',
            batch_size=64,
            batch_labels=None,
            #correct='all', #'missing'
            layers=[]
        ):

        if not batch_labels:
            batch_labels = self.batch_labels

        if len(layers) == 0:
            layers = [[None]*len(modality_adata) for i, modality_adata in enumerate(adatas)]

        pair_count = self.prep_paired_groups(pair_groups)

        # TODO: check if need unique_pairs_of_modalities
        self.model.paired_dict = self.pair_groups_dict
        #self.model.unique_pairs_of_modalities = self.unique_pairs_of_modalities
        self.model.modalities_per_group = self.modalities_per_group
        self.model.paired_networks_per_modality_pairs = self.paired_networks_per_modality_pairs

        adatas = self.reshape_adatas(adatas, names, layers, pair_groups=pair_groups, batch_labels=batch_labels)
        datasets, _ = self.make_datasets(adatas, val_split=0, modality_key=modality_key, celltype_key=celltype_key, batch_size=batch_size)
        dataloaders = [d.loader for d in datasets]

        ad_integrated, ad_latent, ad_latent_corrected, ad_hs = [], [], [], []

        with torch.no_grad():
            self.model.eval()

            for datas in zip_longest(*dataloaders):
                datas = [data for data in datas if data is not None]
                xs = [data[0].to(self.device) for data in datas]
                names = [data[1] for data in datas]
                modalities = [data[2] for data in datas]
                pair_groups = [data[3] for data in datas]
                celltypes = [data[4] for data in datas]
                indices = [data[5] for data in datas]
                batch_labels = [data[-1] for data in datas]

                group_indices = {}
                for i, pair in enumerate(pair_groups):
                    group_indices[pair] = group_indices.get(pair, []) + [i]

                hs = [self.model.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
                zs = [self.model.bottleneck(h) for h in hs]
                mus = [z[1] for z in zs]
                logvars = [z[2] for z in zs]
                zs = [z[0] for z in zs]
                mus_joint, logvars_joint, joint_pair_groups = self.model.product_of_experts(mus, logvars, pair_groups)
                zs_joint = [self.model.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
                zs_corrected, new_modalities, new_pair_groups, new_batch_labels, xs, size_factors = self.model.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels)

                for i, (z_corrected, pair, mod) in enumerate(zip(zs_corrected, new_pair_groups, new_modalities)):
                    z = sc.AnnData(z_corrected.detach().cpu().numpy())
                    z.obs['modality'] = mod
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs['study'] = pair
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    ad_latent_corrected.append(z)

                for i, (z_joint, pair) in enumerate(zip(zs_joint, joint_pair_groups)):
                    z = sc.AnnData(z_joint.detach().cpu().numpy())
                    mods = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(mods)
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs['study'] = pair
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    ad_integrated.append(z)

                for h, z_latent, pg, mod, cell_type, idx in zip(hs, zs, pair_groups, modalities, celltypes, indices):
                    z = sc.AnnData(z_latent.detach().cpu().numpy())
                    z.obs['modality'] = mod
                    z.obs['barcode'] = idx
                    z.obs['study'] = pg
                    z.obs[celltype_key] = cell_type
                    ad_latent.append(z)
                    if len(self.modalities_per_group[pg]) == 1:
                        ad_integrated.append(z)

                    z = sc.AnnData(h.detach().cpu().numpy())
                    z.obs['modality'] = mod
                    z.obs['barcode'] = idx
                    z.obs['study'] = pg
                    z.obs[celltype_key] = cell_type
                    ad_hs.append(z)

        return sc.AnnData.concatenate(*ad_integrated), sc.AnnData.concatenate(*ad_latent), sc.AnnData.concatenate(*ad_latent_corrected), sc.AnnData.concatenate(*ad_hs)

    def calc_integ_loss(self, zs, pair_groups, correct, kernel_type, version, zs_not_corrected=None):
        loss = 0
        for zi, pgi in zip(zs, pair_groups):
            for zj, pgj in zip(zs, pair_groups):
                if pgi == pgj:  # do not integrate one dataset with itself
                    continue
                loss += MMD(kernel_type=kernel_type)(zi, zj)
        return loss

    def impute_batch(self, x, pair, mod, batch, target_pair, target_modality):
        h = self.model.to_shared_dim(x, mod, batch)
        z = self.model.bottleneck(h)
        mus = z[1]
        logvars = z[2]
        z = z[0]
        # fix batch label, so it takes mean of available batches
        r = self.model.decode_from_shared(z, target_modality, pair, 0)
        return r
