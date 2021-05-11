import torch
from torch import nn
from .multivae_poe_small import MultiVAE_PoE_small, MultiVAETorch_PoE_small
from .losses import MMD

class MultiVAETorch_PoE_small_mse(MultiVAETorch_PoE_small):
    def forward(self, xs, modalities, pair_groups, batch_labels, size_factors):
        hs = [self.to_shared_dim(x, mod, batch_label) for x, mod, batch_label in zip(xs, modalities, batch_labels)]
        zs = [self.bottleneck(h) for h in hs]
        mus = [z[1] for z in zs]
        logvars = [z[2] for z in zs]
        zs = [z[0] for z in zs]
        mus_joint, logvars_joint, _ = self.product_of_experts(mus, logvars, pair_groups)
        zs_joint = [self.reparameterize(mu_joint, logvar_joint) for mu_joint, logvar_joint in zip(mus_joint, logvars_joint)]
        out = self.prep_latent(xs, zs, zs_joint, modalities, pair_groups, batch_labels)
        zs_corrected = out[0]
        rs = [self.decode_from_shared(z, mod, pair_group, batch_label) for z, mod, pair_group, batch_label in zip(zs, modalities, pair_groups, batch_labels)]
        return rs, zs_corrected, mus+mus_joint, logvars+logvars_joint, pair_groups, modalities, batch_labels, xs, size_factors, zs

class MultiVAE_PoE_small_mse(MultiVAE_PoE_small):
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

        self.model = MultiVAETorch_PoE_small_mse(self.encoders, self.decoders, self.shared_encoder, self.shared_decoder,
                                   self.mu, self.logvar, self.modality_vecs, self.device, self.condition, self.n_batch_labels,
                                   self.pair_groups_dict, self.modalities_per_group, self.paired_networks_per_modality_pairs)

    def calc_integ_loss(self, zs, pair_groups, correct, kernel_type, version, zs_not_corrected):
        loss = 0
        for i, (zi, pgi) in enumerate(zip(zs, pair_groups)):
            # calculate MSE between z_i and z_joint + v_mod
            if len(self.model.modalities_per_group[pgi]) > 1:
                loss += nn.MSELoss()(zi, zs_not_corrected[i])
            for zj, pgj in zip(zs, pair_groups):
                if pgi == pgj:  # do not integrate one dataset with itself
                    continue
                loss += MMD(kernel_type=kernel_type)(zi, zj)
        return loss
