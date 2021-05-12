import torch
import scanpy as sc
import numpy as np
from .multivae import MultiVAE, MultiVAETorch
from .mlp_decoder import MLP_decoder
from .losses import MMD
from itertools import zip_longest

class MultiVAETorch_smaller(MultiVAETorch):
    def forward(self, xs, modalities, pair_groups, batch_labels, size_factors):
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        hs_concat, concat_pair_groups = self.encode_pairs(hs, pair_groups)
        zs = [self.bottleneck(h) for h in hs_concat]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        zs = self.prep_latent(zs, concat_pair_groups)
        rs = [self.decode_from_shared(z, mod, pair_group, batch_label) for z, mod, pair_group, batch_label in zip(zs, modalities, pair_groups, batch_labels)]
        return rs, zs, mus, logvars, concat_pair_groups

    # TODO: matrix form? store in a dict
    def prep_latent(self, zs, pair_groups):
        zs_corrected = []
        for z, pg in zip(zs, pair_groups):
            group_size = len(self.modalities_per_group[pg])
            if group_size > 1:
                mask = torch.zeros(group_size, self.n_modality).to(self.device)
                for i, mod in enumerate(self.modalities_per_group[pg]):
                    mask[:, mod] = -torch.ones(group_size).to(self.device)
                    mask[:, mod] += torch.eye(group_size)[i].to(self.device)
                mod_vecs = mask @ self.modality_vectors.weight
                mod_vecs = [mod_vecs[i] for i, _ in enumerate(mod_vecs)]
                z_group = [z + mod_vec for mod_vec in mod_vecs]
                zs_corrected.extend(z_group)
            else:
                zs_corrected.append(z)
        return zs_corrected

class MultiVAE_smaller(MultiVAE):
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

        self.decoders = [MLP_decoder(z_dim + self.n_batch_labels[i], x_dim, hs[::-1], output_activation=out_act,
                             dropout=dropout, norm=normalization, loss=loss) if x_dim > 0 else None for i, (x_dim, hs, out_act, loss) in enumerate(zip(self.x_dims, hiddens, output_activations, self.losses))]
        self.model = MultiVAETorch_smaller(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.device, self.condition, self.n_batch_labels,
                                   self.pair_groups_dict, self.modalities_per_group, self.paired_networks_per_modality_pairs)

    def calc_integ_loss(self, zs, pair_groups, correct, kernel_type, version, zs_not_corrected=None):
        loss = 0
        for zi, pgi in zip(zs, pair_groups):
            for zj, pgj in zip(zs, pair_groups):
                if pgi == pgj:  # do not integrate one dataset with itself
                    continue
                loss += MMD(kernel_type=kernel_type)(zi, zj)
        return loss

    def test(self,
            adatas,
            names,
            pair_groups,
            modality_key='modality',
            celltype_key='cell_type',
            batch_size=64,
            batch_labels=None,
            version='1', # '1' or '2'
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
                hs_concat, new_pair_groups = self.model.encode_pairs(hs, pair_groups)

                zs = [self.model.bottleneck(h) for h in hs_concat]

                mus = [z[1] for z in zs]
                logvars = [z[2] for z in zs]
                zs_latent = [z[0] for z in zs]

                zs_corrected = self.model.prep_latent(zs_latent, new_pair_groups)

                for i, (z_latent, pair) in enumerate(zip(zs_latent, new_pair_groups)):
                    z = sc.AnnData(z_latent.detach().cpu().numpy())
                    mods = np.array(names)[group_indices[pair], ]
                    z.obs['modality'] = '-'.join(mods)
                    z.obs['barcode'] = list(indices[group_indices[pair][0]])
                    z.obs['study'] = pair
                    z.obs[celltype_key] = celltypes[group_indices[pair][0]]
                    ad_latent.append(z)

                for h, z_corrected, pg, mod, cell_type, idx in zip(hs, zs_corrected, pair_groups, modalities, celltypes, indices):
                    z = sc.AnnData(z_corrected.detach().cpu().numpy())
                    z.obs['modality'] = mod
                    z.obs['barcode'] = idx
                    z.obs['study'] = pg
                    z.obs[celltype_key] = cell_type
                    ad_latent_corrected.append(z)


                    z = sc.AnnData(h.detach().cpu().numpy())
                    z.obs['modality'] = mod
                    z.obs['barcode'] = idx
                    z.obs['study'] = pg
                    z.obs[celltype_key] = cell_type
                    ad_hs.append(z)
        # integrated = latent
        return sc.AnnData.concatenate(*ad_latent), sc.AnnData.concatenate(*ad_latent), sc.AnnData.concatenate(*ad_latent_corrected), sc.AnnData.concatenate(*ad_hs)

    def impute_batch(self, x, pair, mod, batch, target_pair, target_modality):
        h = self.model.to_shared_dim(x, mod, batch)
        hs_concat, new_pair_group = self.model.encode_pairs([h], [pair])
        # fix [0]
        z = self.model.bottleneck(hs_concat[0])
        mus = z[1]
        logvars = z[2]
        z = z[0]
        # fix batch label, so it takes mean of available batches
        r = self.model.decode_from_shared(z, target_modality, target_pair, 0)
        return r
