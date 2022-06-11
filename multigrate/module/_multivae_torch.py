import torch

from torch import nn
from torch.nn import functional as F
from ..distributions import *
from ..nn import *
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi import _CONSTANTS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


class MultiVAETorch(BaseModuleClass):
    def __init__(
        self,
        modality_lengths,
        condition_encoders=False,
        condition_decoders=True,
        normalization="layer",
        z_dim=15,
        h_dim=32,
        losses=[],
        dropout=0.2,
        cond_dim=10,
        kernel_type="gaussian",
        loss_coefs=[],
        num_groups=1,  # to integrate on
        integrate_on_idx=None,
        cat_covariate_dims=[],
        cont_covariate_dims=[],
        cont_cov_type="logsigm",
        n_layers_cont_embed: int = 1,
        n_layers_encoders=[],
        n_layers_decoders=[],
        n_layers_shared_decoder: int = 1,
        n_hidden_cont_embed: int = 32,
        n_hidden_encoders=[],
        n_hidden_decoders=[],
        n_hidden_shared_decoder: int = 32,
        add_shared_decoder=True,
        mmd="latent",  # or both or marginal
    ):
        super().__init__()

        self.input_dims = modality_lengths
        self.condition_encoders = condition_encoders
        self.condition_decoders = condition_decoders
        self.n_modality = len(self.input_dims)
        self.kernel_type = kernel_type
        self.integrate_on_idx = integrate_on_idx
        self.add_shared_decoder = add_shared_decoder
        self.n_cont_cov = len(cont_covariate_dims)
        self.cont_cov_type = cont_cov_type
        self.mmd = mmd

        # needed to query to reference
        self.normalization = normalization
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.kernel_type = kernel_type
        self.n_layers_cont_embed = n_layers_cont_embed
        self.n_layers_encoders = n_layers_encoders
        self.n_layers_decoders = n_layers_decoders
        self.n_layers_shared_decoder = n_layers_shared_decoder
        self.n_hidden_cont_embed = n_hidden_cont_embed
        self.n_hidden_encoders = n_hidden_encoders
        self.n_hidden_decoders = n_hidden_decoders
        self.n_hidden_shared_decoder = n_hidden_shared_decoder

        # TODO: clean
        if len(losses) == 0:
            self.losses = ["mse"] * self.n_modality
        elif len(losses) == self.n_modality:
            self.losses = losses
        else:
            raise ValueError(
                f"losses has to be the same length as the number of modalities. number of modalities = {self.n_modality} != {len(losses)} = len(losses)"
            )

        # TODO: add warning that using these
        if len(n_layers_encoders) == 0:
            n_layers_encoders = [2] * self.n_modality
        if len(n_layers_decoders) == 0:
            n_layers_decoders = [2] * self.n_modality
        if len(n_hidden_encoders) == 0:
            n_hidden_encoders = [128] * self.n_modality
        if len(n_hidden_decoders) == 0:
            n_hidden_decoders = [128] * self.n_modality

        self.loss_coefs = {
            "recon": 1,
            "kl": 1e-6,
            "integ": 1e-2,
            "cycle": 0,
        }
        for i in range(self.n_modality):
            self.loss_coefs[str(i)] = 1
        self.loss_coefs.update(loss_coefs)

        # assume for now that can only use nb/zinb once, i.e. for RNA-seq modality
        # TODO: add check for multiple nb/zinb losses given
        self.theta = None
        for i, loss in enumerate(losses):
            if loss in ["nb", "zinb"]:
                self.theta = torch.nn.Parameter(
                    torch.randn(self.input_dims[i], num_groups)
                )
                # self.register_parameter(name='theta', param=self.theta)
                break

        # shared decoder
        if self.add_shared_decoder:
            self.shared_decoder = MLP(
                n_input=z_dim + self.n_modality,
                n_output=h_dim,
                n_layers=n_layers_shared_decoder,
                n_hidden=n_hidden_shared_decoder,
                dropout_rate=dropout,
                normalization=normalization,
            )
        # modality encoders
        cond_dim_enc = (
            cond_dim * (len(cat_covariate_dims) + len(cont_covariate_dims))
            if self.condition_encoders
            else 0
        )
        self.encoders = [
            MLP(
                n_input=x_dim + cond_dim_enc,
                n_output=z_dim,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout,
                normalization=normalization,
            )
            for x_dim, n_layers, n_hidden in zip(
                self.input_dims, n_layers_encoders, n_hidden_encoders
            )
        ]
        # modality decoders
        cond_dim_dec = (
            cond_dim * (len(cat_covariate_dims) + len(cont_covariate_dims))
            if self.condition_decoders
            else 0
        )
        dec_input = h_dim if self.add_shared_decoder else z_dim

        self.decoders = [
            Decoder(
                n_input=dec_input + cond_dim_dec,
                n_output=x_dim,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout,
                normalization=normalization,
                loss=loss,
            )
            for x_dim, loss, n_layers, n_hidden in zip(
                self.input_dims, self.losses, n_layers_decoders, n_hidden_decoders
            )
        ]

        self.mus = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]
        self.logvars = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]

        self.cat_covariate_embeddings = [
            nn.Embedding(dim, cond_dim) for dim in cat_covariate_dims
        ]
        self.cont_covariate_embeddings = nn.Embedding(self.n_cont_cov, cond_dim)

        if self.cont_cov_type == "mlp":
            self.cont_covariate_curves = torch.nn.ModuleList()
            for _ in range(self.n_cont_cov):
                n_input = n_hidden_cont_embed if n_layers_cont_embed > 1 else 1
                self.cont_covariate_curves.append(
                    nn.Sequential(
                        MLP(
                            n_input=1,
                            n_output=n_hidden_cont_embed,
                            n_layers=n_layers_cont_embed - 1,
                            n_hidden=n_hidden_cont_embed,
                            dropout_rate=dropout,
                            normalization=normalization,
                        ),
                        nn.Linear(n_input, 1),
                    )
                    if n_layers_cont_embed > 1
                    else nn.Linear(n_input, 1)
                )
        else:
            self.cont_covariate_curves = GeneralizedSigmoid(
                dim=self.n_cont_cov,
                # device=self.device,
                nonlin=self.cont_cov_type,
            )

        # self.cont_covariate_embeddings = [nn.Linear(dim, cond_dim) for dim in cont_covariate_dims] # dim is always 1 here

        # register sub-modules
        for i, (enc, dec, mu, logvar) in enumerate(
            zip(self.encoders, self.decoders, self.mus, self.logvars)
        ):
            self.add_module(f"encoder_{i}", enc)
            self.add_module(f"decoder_{i}", dec)
            self.add_module(f"mu_{i}", mu)
            self.add_module(f"logvar_{i}", logvar)

        for i, emb in enumerate(self.cat_covariate_embeddings):
            self.add_module(f"cat_covariate_embedding_{i}", emb)

        ##     self.add_module(f'cont_covariate_embedding_{i}', emb)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, z, i):
        mu = self.mus[i](z)
        logvar = self.logvars[i](z)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def x_to_h(self, x, i):
        return self.encoders[i](x)

    def h_to_x(self, h, i):
        x = self.decoders[i](h)
        return x

    def z_to_h(self, z, mod):
        z = torch.stack([torch.cat((cell, self.modal_vector(mod)[0])) for cell in z])
        h = self.shared_decoder(z)
        return h

    def modal_vector(self, i):
        return (
            F.one_hot(torch.tensor([i]).long(), self.n_modality).float().to(self.device)
        )

    def product_of_experts(self, mus, logvars, masks):
        vars = torch.exp(logvars)
        masks = masks.unsqueeze(-1).repeat(1, 1, vars.shape[-1])
        mus_joint = torch.sum(mus * masks / vars, dim=1)
        logvars_joint = torch.ones_like(mus_joint)  # batch size
        logvars_joint += torch.sum(masks / vars, dim=1)
        logvars_joint = 1.0 / logvars_joint  # inverse
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)
        return mus_joint, logvars_joint

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(x=x, cat_covs=cat_covs, cont_covs=cont_covs)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z_joint = inference_outputs["z_joint"]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return dict(z_joint=z_joint, cat_covs=cat_covs, cont_covs=cont_covs)

    @auto_move_data
    def inference(self, x, cat_covs, cont_covs, masks=None):
        # split x into modality xs
        if torch.is_tensor(x):
            xs = torch.split(
                x, self.input_dims, dim=-1
            )  # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        else:
            xs = x

        if masks is None:
            masks = [x.sum(dim=1) > 0 for x in xs]  # list of masks per modality
            masks = torch.stack(masks, dim=1)

        # if we want to condition encoders, i.e. concat covariates to the input
        if self.condition_encoders:
            # check if need to concat categorical covariates
            if cat_covs is not None:
                cat_embedds = [
                    cat_covariate_embedding(covariate.long())
                    for cat_covariate_embedding, covariate in zip(
                        self.cat_covariate_embeddings, cat_covs.T
                    )
                ]
            else:
                cat_embedds = []

            if len(cat_embedds) > 0:
                cat_embedds = torch.cat(cat_embedds, dim=-1)
            else:
                cat_embedds = torch.Tensor().to(self.device)
            # check if need to concat continuous covariates
            if self.n_cont_cov > 0:
                if cont_covs.shape[-1] != self.n_cont_cov:  # get rid of size_factors
                    cont_covs = cont_covs[:, 0 : self.n_cont_cov]
                cont_embedds = self.compute_cont_cov_embeddings_(cont_covs)
            else:
                cont_embedds = torch.Tensor().to(self.device)
            # concatenate input with categorical and continuous covariates
            xs = [
                torch.cat([x, cat_embedds, cont_embedds], dim=-1) for x in xs
            ]  # concat embedding to each modality x along the feature axis

        # TODO don't forward if mask is 0 for that dataset for that modality
        # hs = hidden state that we get after the encoder but before calculating mu and logvar for each modality
        hs = [self.x_to_h(x, mod) for mod, x in enumerate(xs)]
        # out = [zs_marginal, mus, logvars] and len(zs_marginal) = len(mus) = len(logvars) = number of modalities
        out = [self.bottleneck(h, mod) for mod, h in enumerate(hs)]
        # split out into zs_marginal, mus and logvars TODO check if easier to use split
        zs_marginal = [mod_out[0] for mod_out in out]
        z_marginal = torch.stack(zs_marginal, dim=1)
        mus = [mod_out[1] for mod_out in out]
        mu = torch.stack(mus, dim=1)
        logvars = [mod_out[2] for mod_out in out]
        logvar = torch.stack(logvars, dim=1)
        mu_joint, logvar_joint = self.product_of_experts(mu, logvar, masks)
        z_joint = self.reparameterize(mu_joint, logvar_joint)
        # drop mus and logvars according to masks for kl calculation
        # TODO here or in loss calculation? check multiVI
        # return mus+mus_joint
        return dict(
            z_joint=z_joint, mu=mu_joint, logvar=logvar_joint, z_marginal=z_marginal
        )

    @auto_move_data
    def generative(self, z_joint, cat_covs, cont_covs):
        z = z_joint.unsqueeze(1).repeat(1, self.n_modality, 1)

        if self.add_shared_decoder:
            mod_vecs = self.modal_vector(
                list(range(self.n_modality))
            )  # shape 1 x n_mod x n_mod
            mod_vecs = mod_vecs.repeat(
                z.shape[0], 1, 1
            )  # shape batch_size x n_mod x n_mod
            z = torch.cat(
                [z, mod_vecs], dim=-1
            )  # shape batch_size x n_mod x latent_dim+n_mod
            z = self.shared_decoder(z)

        zs = torch.split(z, 1, dim=1)

        if self.condition_decoders:
            if cat_covs is not None:
                cat_embedds = [
                    cat_covariate_embedding(covariate.long())
                    for cat_covariate_embedding, covariate in zip(
                        self.cat_covariate_embeddings, cat_covs.T
                    )
                ]
            else:
                cat_embedds = []
            if len(cat_embedds) > 0:
                cat_embedds = torch.cat(cat_embedds, dim=-1)
            else:
                cat_embedds = torch.Tensor().to(self.device)

            if self.n_cont_cov > 0:
                if cont_covs.shape[-1] != self.n_cont_cov:  # get rid of size_factors
                    # raise RuntimeError("cont_covs.shape[-1] != self.n_cont_cov") # still can happen when query to ref
                    cont_covs = cont_covs[:, 0 : self.n_cont_cov]
                cont_embedds = self.compute_cont_cov_embeddings_(cont_covs)
            else:
                cont_embedds = torch.Tensor().to(self.device)

            zs = [
                torch.cat([z.squeeze(1), cat_embedds, cont_embedds], dim=-1) for z in zs
            ]  # concat embedding to each modality x along the feature axis

        rs = [self.h_to_x(z, mod) for mod, z in enumerate(zs)]
        return dict(rs=rs)

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):

        x = tensors[_CONSTANTS.X_KEY]
        if self.integrate_on_idx is not None:
            integrate_on = tensors.get(_CONSTANTS.CAT_COVS_KEY)[
                :, self.integrate_on_idx
            ]
        else:
            integrate_on = torch.zeros(x.shape[0], 1).to(self.device)

        size_factor = tensors.get(_CONSTANTS.CONT_COVS_KEY)
        if size_factor is not None:
            size_factor = size_factor[:, -1]  # always last
        rs = generative_outputs["rs"]
        mu = inference_outputs["mu"]
        logvar = inference_outputs["logvar"]
        z_joint = inference_outputs["z_joint"]
        z_marginal = inference_outputs[
            "z_marginal"
        ]  # batch_size x n_modalities x latent_dim

        xs = torch.split(
            x, self.input_dims, dim=-1
        )  # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        masks = [x.sum(dim=1) > 0 for x in xs]  # [batch_size] * num_modalities

        recon_loss, modality_recon_losses = self.calc_recon_loss(
            xs, rs, self.losses, integrate_on, size_factor, self.loss_coefs, masks
        )
        kl_loss = kl(Normal(mu, torch.sqrt(torch.exp(logvar))), Normal(0, 1)).sum(dim=1)

        if self.loss_coefs["integ"] == 0:
            integ_loss = torch.tensor(0.0).to(self.device)
        else:
            integ_loss = torch.tensor(0.0).to(self.device)
            if self.mmd == "latent" or self.mmd == "both":
                integ_loss += self.calc_integ_loss(z_joint, integrate_on).to(
                    self.device
                )
            if self.mmd == "marginal" or self.mmd == "both":
                for i in range(len(masks)):
                    for j in range(i + 1, len(masks)):
                        idx_where_to_calc_mmd = torch.eq(
                            masks[i] == masks[j],
                            torch.eq(masks[i], torch.ones_like(masks[i])),
                        )
                        if (
                            idx_where_to_calc_mmd.any()
                        ):  # if need to calc mmd for a group between modalities
                            marginal_i = z_marginal[:, i, :][idx_where_to_calc_mmd]
                            marginal_j = z_marginal[:, j, :][idx_where_to_calc_mmd]
                            marginals = torch.cat([marginal_i, marginal_j])
                            modalities = torch.cat(
                                [
                                    torch.Tensor([i] * marginal_i.shape[0]),
                                    torch.Tensor([j] * marginal_j.shape[0]),
                                ]
                            ).to(self.device)

                            integ_loss += self.calc_integ_loss(
                                marginals, modalities
                            ).to(self.device)

                for i in range(len(masks)):
                    marginal_i = z_marginal[:, i, :]
                    marginal_i = marginal_i[masks[i]]
                    group_marginal = integrate_on[masks[i]]
                    integ_loss += self.calc_integ_loss(marginal_i, group_marginal).to(
                        self.device
                    )

        cycle_loss = (
            torch.tensor(0.0).to(self.device)
            if self.loss_coefs["cycle"] == 0
            else self.calc_cycle_loss(
                xs,
                z_joint,
                integrate_on,
                masks,
                self.losses,
                size_factor,
                self.loss_coefs,
            )
        )

        loss = torch.mean(
            self.loss_coefs["recon"] * recon_loss
            + self.loss_coefs["kl"] * kl_loss
            + self.loss_coefs["integ"] * integ_loss
            + self.loss_coefs["cycle"] * cycle_loss
        )
        reconst_losses = dict(recon_loss=recon_loss)
        modality_recon_losses = {
            i: modality_recon_losses[i] for i in range(len(modality_recon_losses))
        }

        return LossRecorder(
            loss,
            reconst_losses,
            kl_loss,
            kl_global=torch.tensor(0.0),
            integ_loss=integ_loss,
            cycle_loss=cycle_loss,
            modality_recon_losses=modality_recon_losses,
        )

    # TODO ??
    @torch.no_grad()
    def sample(self, tensors):
        with torch.no_grad():
            (
                _,
                generative_outputs,
            ) = self.forward(tensors, compute_loss=False)

        return generative_outputs["rs"]

    def calc_recon_loss(self, xs, rs, losses, group, size_factor, loss_coefs, masks):
        loss = []
        for i, (x, r, loss_type) in enumerate(zip(xs, rs, losses)):
            if len(r) != 2 and len(r.shape) == 3:
                r = r.squeeze()
            if loss_type == "mse":
                mse_loss = loss_coefs[str(i)] * torch.sum(
                    nn.MSELoss(reduction="none")(r, x), dim=-1
                )
                loss.append(mse_loss)
            elif loss_type == "nb":
                dec_mean = r
                size_factor_view = size_factor.unsqueeze(1).expand(
                    dec_mean.size(0), dec_mean.size(1)
                )
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[group.squeeze().long()]
                dispersion = torch.exp(dispersion)
                nb_loss = torch.sum(
                    NegativeBinomial(mu=dec_mean, theta=dispersion).log_prob(x), dim=-1
                )
                nb_loss = loss_coefs[str(i)] * nb_loss
                loss.append(-nb_loss)
            elif loss_type == "zinb":
                dec_mean, dec_dropout = r
                dec_mean = dec_mean.squeeze()
                dec_dropout = dec_dropout.squeeze()
                size_factor_view = size_factor.unsqueeze(1).expand(
                    dec_mean.size(0), dec_mean.size(1)
                )
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[group.squeeze().long()]
                dispersion = torch.exp(dispersion)
                zinb_loss = torch.sum(
                    ZeroInflatedNegativeBinomial(
                        mu=dec_mean, theta=dispersion, zi_logits=dec_dropout
                    ).log_prob(x),
                    dim=-1,
                )
                zinb_loss = loss_coefs[str(i)] * zinb_loss
                loss.append(-zinb_loss)
            elif loss_type == "bce":
                bce_loss = loss_coefs[str(i)] * torch.sum(
                    torch.nn.BCELoss(reduction="none")(r, x), dim=-1
                )
                loss.append(bce_loss)

        return (
            torch.sum(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=1),
            torch.sum(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=0),
        )

    def calc_integ_loss(self, z, group):
        loss = torch.tensor(0.0).to(self.device)
        unique = torch.unique(group)
        if len(unique) > 1:
            zs = [z[group == i] for i in unique]
            for i in range(len(zs)):
                for j in range(i + 1, len(zs)):
                    loss += MMD(kernel_type=self.kernel_type)(zs[i], zs[j])
        return loss

    def calc_cycle_loss(
        self, xs, z, cat_covs, cont_covs, masks, losses, size_factor, loss_coefs
    ):
        generative_outputs = self.generative(z, cat_covs, cont_covs)
        rs = generative_outputs["rs"]
        for i, r in enumerate(rs):
            if len(r) == 2:  # hack for zinb
                rs[i] = r[0]
            rs[i] = rs[i].squeeze()

        masks_stacked = torch.stack(masks, dim=1)
        complement_masks = torch.logical_not(masks_stacked)

        inference_outputs = self.inference(rs, cat_covs, cont_covs, complement_masks)
        z_joint = inference_outputs["z_joint"]
        # generate again
        generative_outputs = self.generative(z_joint, cat_covs, cont_covs)
        rs = generative_outputs["rs"]

        group = cat_covs[:, self.integrate_on_idx]

        return self.calc_recon_loss(
            xs, rs, losses, group, size_factor, loss_coefs, masks
        )

    def compute_cont_cov_embeddings_(self, covs):
        """Adapted from
        Title: CPA (c) Facebook, Inc.
        Date: 26.01.2022
        Link to the used code:
        https://github.com/facebookresearch/CPA/blob/382ff641c588820a453d801e5d0e5bb56642f282/compert/model.py#L342
        """
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.
        """
        if self.cont_cov_type == "mlp":
            embeddings = []
            for cov in range(covs.size(1)):
                this_cov = covs[:, cov].view(-1, 1)
                embeddings.append(
                    self.cont_covariate_curves[cov](this_cov).sigmoid()
                )  # * this_drug.gt(0)) # TODO check what this .gt(0) is
            return torch.cat(embeddings, 1) @ self.cont_covariate_embeddings.weight
        else:
            return (
                self.cont_covariate_curves(covs) @ self.cont_covariate_embeddings.weight
            )
