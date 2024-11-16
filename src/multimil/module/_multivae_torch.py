import warnings
from typing import Literal

import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from multimil.distributions import MMD, Jeffreys
from multimil.nn import MLP, Decoder, GeneralizedSigmoid


class MultiVAETorch(BaseModuleClass):
    """MultiMIL's multimodal integration module.

    Parameters
    ----------
    modality_lengths
        List with lengths of each modality.
    condition_encoders
        Boolean to indicate if to condition encoders.
    condition_decoders
        Boolean to indicate if to condition decoders.
    normalization
        One of the following
        * ``'layer'`` - layer normalization
        * ``'batch'`` - batch normalization
        * ``None`` - no normalization.
    z_dim
        Dimensionality of the latent space.
    losses
        List of which losses to use. For each modality can be one of the following:
        * ``'mse'`` - mean squared error
        * ``'nb'`` - negative binomial
        * ``zinb`` - zero-inflated negative binomial
        * ``bce`` - binary cross-entropy.
    dropout
        Dropout rate for neural networks.
    cond_dim
        Dimensionality of the covariate embeddings.
    kernel_type
        One of the following:
        * ``'gaussian'`` - Gaussian kernel
        * ``'not gaussian'`` - not Gaussian kernel.
    loss_coefs
        Dictionary with weights for each of the losses.
    num_groups
        Number of groups to integrate on.
    integrate_on_idx
        Indices on which to integrate on.
    cat_covariate_dims
        List with number of classes for each of the categorical covariates.
    cont_covariate_dims
        List of 1's for each of the continuous covariate.
    cont_cov_type
        How to transform continuous covariate before multiplying with the embedding. One of the following:
        * ``'logsim'`` - generalized sigmoid
        * ``'mlp'`` - MLP.
    n_layers_cont_embed
        Number of layers for the transformation of the continuous covariates before multiplying with the embedding.
    n_hidden_cont_embed
        Number of nodes in hidden layers in the network that transforms continuous covariates.
    n_layers_encoders
        Number of layers in each encoder.
    n_layers_decoders
        Number of layers in each decoder.
    n_hidden_encoders
        Number of nodes in hidden layers in encoders.
    n_hidden_decoders
        Number of nodes in hidden layers in decoders.
    alignment_type
        How to calculate integration loss. One of the following
        * ``'latent'`` - only on the latent representations
        * ``'marginal'`` - only on the marginal representations
        * ``both`` - the sum of the two above.
    """

    def __init__(
        self,
        modality_lengths,
        condition_encoders=False,
        condition_decoders=True,
        normalization: Literal["layer", "batch", None] = "layer",
        z_dim=16,
        losses=None,
        dropout=0.2,
        cond_dim=16,
        kernel_type="gaussian",
        loss_coefs=None,
        num_groups=1,
        integrate_on_idx=None,
        cat_covariate_dims=None,
        cont_covariate_dims=None,
        cat_covs_idx=None,
        cont_covs_idx=None,
        cont_cov_type="logsigm",
        n_layers_cont_embed: int = 1,
        n_layers_encoders=None,
        n_layers_decoders=None,
        n_hidden_cont_embed: int = 16,
        n_hidden_encoders=None,
        n_hidden_decoders=None,
        modality_alignment=None,
        alignment_type="latent",
        activation="leaky_relu",
        initialization=None,
        mix="product",
    ):
        super().__init__()

        self.input_dims = modality_lengths
        self.condition_encoders = condition_encoders
        self.condition_decoders = condition_decoders
        self.n_modality = len(self.input_dims)
        self.kernel_type = kernel_type
        self.integrate_on_idx = integrate_on_idx
        self.n_cont_cov = len(cont_covariate_dims)
        self.cont_cov_type = cont_cov_type
        self.alignment_type = alignment_type
        self.modality_alignment = modality_alignment
        self.normalization = normalization
        self.z_dim = z_dim
        self.dropout = dropout
        self.cond_dim = cond_dim
        self.kernel_type = kernel_type
        self.n_layers_cont_embed = n_layers_cont_embed
        self.n_layers_encoders = n_layers_encoders
        self.n_layers_decoders = n_layers_decoders
        self.n_hidden_cont_embed = n_hidden_cont_embed
        self.n_hidden_encoders = n_hidden_encoders
        self.n_hidden_decoders = n_hidden_decoders
        self.cat_covs_idx = cat_covs_idx
        self.cont_covs_idx = cont_covs_idx
        self.mix = mix

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        else:
            raise NotImplementedError(
                f'activation should be one of ["leaky_relu", "tanh"], but activation={activation} was passed.'
            )

        # TODO: add warnings that mse is used
        if losses is None:
            self.losses = ["mse"] * self.n_modality
        elif len(losses) == self.n_modality:
            self.losses = losses
        else:
            raise ValueError(
                f"losses has to be the same length as the number of modalities. number of modalities = {self.n_modality} != {len(losses)} = len(losses)"
            )
        if cat_covariate_dims is None:
            raise ValueError("cat_covariate_dims = None was passed.")
        if cont_covariate_dims is None:
            raise ValueError("cont_covariate_dims = None was passed.")

        # TODO: add warning that using these
        if self.n_layers_encoders is None:
            self.n_layers_encoders = [2] * self.n_modality
        if self.n_layers_decoders is None:
            self.n_layers_decoders = [2] * self.n_modality
        if self.n_hidden_encoders is None:
            self.n_hidden_encoders = [128] * self.n_modality
        if self.n_hidden_decoders is None:
            self.n_hidden_decoders = [128] * self.n_modality

        self.loss_coefs = {
            "recon": 1,
            "kl": 1e-6,
            "integ": 0,
        }
        for i in range(self.n_modality):
            self.loss_coefs[str(i)] = 1
        if loss_coefs is not None:
            self.loss_coefs.update(loss_coefs)

        # assume for now that can only use nb/zinb once, i.e. for RNA-seq modality
        # TODO: add check for multiple nb/zinb losses given
        self.theta = None
        for i, loss in enumerate(losses):
            if loss in ["nb", "zinb"]:
                self.theta = torch.nn.Parameter(torch.randn(self.input_dims[i], num_groups))
                break

        # modality encoders
        cond_dim_enc = cond_dim * (len(cat_covariate_dims) + len(cont_covariate_dims)) if self.condition_encoders else 0
        self.encoders = [
            MLP(
                n_input=x_dim + cond_dim_enc,
                n_output=z_dim,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout,
                normalization=normalization,
                activation=self.activation,
            )
            for x_dim, n_layers, n_hidden in zip(
                self.input_dims, self.n_layers_encoders, self.n_hidden_encoders, strict=False
            )
        ]

        # modality decoders
        cond_dim_dec = cond_dim * (len(cat_covariate_dims) + len(cont_covariate_dims)) if self.condition_decoders else 0
        dec_input = z_dim
        self.decoders = [
            Decoder(
                n_input=dec_input + cond_dim_dec,
                n_output=x_dim,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout,
                normalization=normalization,
                activation=self.activation,
                loss=loss,
            )
            for x_dim, loss, n_layers, n_hidden in zip(
                self.input_dims, self.losses, self.n_layers_decoders, self.n_hidden_decoders, strict=False
            )
        ]

        self.mus = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]
        self.logvars = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]

        self.cat_covariate_embeddings = [nn.Embedding(dim, cond_dim) for dim in cat_covariate_dims]
        if self.n_cont_cov > 0:
            self.cont_covariate_embeddings = nn.Embedding(self.n_cont_cov, cond_dim)
            if self.cont_cov_type == "mlp":
                self.cont_covariate_curves = torch.nn.ModuleList()
                for _ in range(self.n_cont_cov):
                    n_input = n_hidden_cont_embed if self.n_layers_cont_embed > 1 else 1
                    self.cont_covariate_curves.append(
                        nn.Sequential(
                            MLP(
                                n_input=1,
                                n_output=n_hidden_cont_embed,
                                n_layers=self.n_layers_cont_embed - 1,
                                n_hidden=n_hidden_cont_embed,
                                dropout_rate=dropout,
                                normalization=normalization,
                                activation=self.activation,
                            ),
                            nn.Linear(n_input, 1),
                        )
                        if self.n_layers_cont_embed > 1
                        else nn.Linear(n_input, 1)
                    )
            else:
                self.cont_covariate_curves = GeneralizedSigmoid(
                    dim=self.n_cont_cov,
                    nonlin=self.cont_cov_type,
                )

        # register sub-modules
        for i, (enc, dec, mu, logvar) in enumerate(
            zip(self.encoders, self.decoders, self.mus, self.logvars, strict=False)
        ):
            self.add_module(f"encoder_{i}", enc)
            self.add_module(f"decoder_{i}", dec)
            self.add_module(f"mu_{i}", mu)
            self.add_module(f"logvar_{i}", logvar)

        for i, emb in enumerate(self.cat_covariate_embeddings):
            self.add_module(f"cat_covariate_embedding_{i}", emb)

        if initialization is not None:
            if initialization == "xavier":
                if activation != "leaky_relu":
                    warnings.warn(
                        f"We recommend using Xavier initialization with leaky_relu, but activation={activation} was passed.",
                        stacklevel=2,
                    )
                for layer in self.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(activation))
            elif initialization == "kaiming":
                if activation != "tanh":
                    warnings.warn(
                        f"We recommend using Kaiming initialization with tanh, but activation={activation} was passed.",
                        stacklevel=2,
                    )
                for layer in self.modules():
                    if isinstance(layer, nn.Linear):
                        # following https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138 (accessed 16.08.22)
                        nn.init.kaiming_normal_(layer.weight, mode="fan_in")

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _bottleneck(self, z, i):
        mu = self.mus[i](z)
        logvar = self.logvars[i](z)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def _x_to_h(self, x, i):
        return self.encoders[i](x)

    def _h_to_x(self, h, i):
        x = self.decoders[i](h)
        return x

    def _product_of_experts(self, mus, logvars, masks):
        vars = torch.exp(logvars)
        masks = masks.unsqueeze(-1).repeat(1, 1, vars.shape[-1])
        mus_joint = torch.sum(mus * masks / vars, dim=1)
        vars_joint = torch.ones_like(mus_joint)  # batch size
        vars_joint += torch.sum(masks / vars, dim=1)
        vars_joint = 1.0 / vars_joint  # inverse
        mus_joint *= vars_joint
        logvars_joint = torch.log(vars_joint)
        return mus_joint, logvars_joint

    def _mixture_of_experts(self, mus, logvars, masks):
        vars = torch.exp(logvars)
        masks = masks.unsqueeze(-1).repeat(1, 1, vars.shape[-1])
        masks = masks.float()
        weights = masks / torch.sum(masks, dim=1, keepdim=True)  # normalize so the sum is 1
        # params of 1/2 * (X + Y)
        mus_mixture = torch.sum(weights * mus, dim=1)
        vars_mixture = torch.sum(weights**2 * vars, dim=1)
        logvars_mixture = torch.log(vars_mixture)
        return mus_mixture, logvars_mixture

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return {"x": x, "cat_covs": cat_covs, "cont_covs": cont_covs}

    def _get_generative_input(self, tensors, inference_outputs):
        z_joint = inference_outputs["z_joint"]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return {"z_joint": z_joint, "cat_covs": cat_covs, "cont_covs": cont_covs}

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        cat_covs: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        masks: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Compute necessary inference quantities.

        Parameters
        ----------
        x
            Tensor of values with shape ``(batch_size, n_input_features)``.
        cat_covs
            Categorical covariates to condition on.
        cont_covs
            Continuous covariates to condition on.
        masks
            List of binary tensors indicating which values in ``x`` belong to which modality.

        Returns
        -------
        Joint representations, marginal representations, joint mu's and logvar's.
        """
        # split x into modality xs
        if torch.is_tensor(x):
            xs = torch.split(
                x, self.input_dims, dim=-1
            )  # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        else:
            xs = x

        # TODO: check if masks still supported
        if masks is None:
            masks = [x.sum(dim=1) > 0 for x in xs]  # list of masks per modality
            masks = torch.stack(masks, dim=1)
        # if we want to condition encoders, i.e. concat covariates to the input
        if self.condition_encoders is True:
            cat_embedds = self._select_cat_covariates(cat_covs)
            cont_embedds = self._select_cont_covariates(cont_covs)

            # concatenate input with categorical and continuous covariates
            xs = [
                torch.cat([x, cat_embedds, cont_embedds], dim=-1) for x in xs
            ]  # concat input to each modality x along the feature axis

        # TODO don't forward if mask is 0 for that dataset for that modality
        # hs = hidden state that we get after the encoder but before calculating mu and logvar for each modality
        hs = [self._x_to_h(x, mod) for mod, x in enumerate(xs)]
        # out = [zs_marginal, mus, logvars] and len(zs_marginal) = len(mus) = len(logvars) = number of modalities
        out = [self._bottleneck(h, mod) for mod, h in enumerate(hs)]
        # split out into zs_marginal, mus and logvars TODO check if easier to use split
        zs_marginal = [mod_out[0] for mod_out in out]
        z_marginal = torch.stack(zs_marginal, dim=1)
        mus = [mod_out[1] for mod_out in out]
        mu = torch.stack(mus, dim=1)
        logvars = [mod_out[2] for mod_out in out]
        logvar = torch.stack(logvars, dim=1)
        if self.mix == "product":
            mu_joint, logvar_joint = self._product_of_experts(mu, logvar, masks)
        elif self.mix == "mixture":
            mu_joint, logvar_joint = self._mixture_of_experts(mu, logvar, masks)
        else:
            raise ValueError(f"mix should be one of ['product', 'mixture'], but mix={self.mix} was passed")
        z_joint = self._reparameterize(mu_joint, logvar_joint)
        # drop mus and logvars according to masks for kl calculation
        # TODO here or in loss calculation? check
        # return mus+mus_joint
        return {
            "z_joint": z_joint,
            "mu": mu_joint,
            "logvar": logvar_joint,
            "z_marginal": z_marginal,
            "mu_marginal": mu,
            "logvar_marginal": logvar,
        }

    @auto_move_data
    def generative(
        self, z_joint: torch.Tensor, cat_covs: torch.Tensor | None = None, cont_covs: torch.Tensor | None = None
    ) -> dict[str, list[torch.Tensor]]:
        """Compute necessary inference quantities.

        Parameters
        ----------
        z_joint
            Tensor of values with shape ``(batch_size, z_dim)``.
        cat_covs
            Categorical covariates to condition on.
        cont_covs
            Continuous covariates to condition on.

        Returns
        -------
        Reconstructed values for each modality.
        """
        z = z_joint.unsqueeze(1).repeat(1, self.n_modality, 1)
        zs = torch.split(z, 1, dim=1)

        if self.condition_decoders is True:
            cat_embedds = self._select_cat_covariates(cat_covs)
            cont_embedds = self._select_cont_covariates(cont_covs)

            zs = [
                torch.cat([z.squeeze(1), cat_embedds, cont_embedds], dim=-1) for z in zs
            ]  # concat embedding to each modality x along the feature axis

        rs = [self._h_to_x(z, mod) for mod, z in enumerate(zs)]
        return {"rs": rs}

    def _select_cat_covariates(self, cat_covs):
        if len(self.cat_covs_idx) > 0:
            cat_covs = torch.index_select(cat_covs, 1, self.cat_covs_idx.to(self.device))
            cat_embedds = [
                cat_covariate_embedding(covariate.long())
                for cat_covariate_embedding, covariate in zip(self.cat_covariate_embeddings, cat_covs.T, strict=False)
            ]
        else:
            cat_embedds = []

        if len(cat_embedds) > 0:
            cat_embedds = torch.cat(cat_embedds, dim=-1)  # TODO check if concatenation is needed
        else:
            cat_embedds = torch.Tensor().to(self.device)
        return cat_embedds

    def _select_cont_covariates(self, cont_covs):
        if len(self.cont_covs_idx) > 0:
            cont_covs = torch.index_select(cont_covs, 1, self.cont_covs_idx.to(self.device))
            if cont_covs.shape[-1] != self.n_cont_cov:  # get rid of size_factors
                cont_covs = cont_covs[:, 0 : self.n_cont_cov]
            cont_embedds = self._compute_cont_cov_embeddings(cont_covs)
        else:
            cont_embedds = torch.Tensor().to(self.device)
        return cont_embedds

    def _calculate_loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        x = tensors[REGISTRY_KEYS.X_KEY]
        if self.integrate_on_idx is not None:
            integrate_on = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)[:, self.integrate_on_idx]
        else:
            integrate_on = torch.zeros(x.shape[0], 1).to(self.device)

        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)

        rs = generative_outputs["rs"]
        mu = inference_outputs["mu"]
        logvar = inference_outputs["logvar"]
        z_joint = inference_outputs["z_joint"]
        z_marginal = inference_outputs["z_marginal"]  # batch_size x n_modalities x latent_dim
        mu_marginal = inference_outputs["mu_marginal"]
        logvar_marginal = inference_outputs["logvar_marginal"]

        xs = torch.split(
            x, self.input_dims, dim=-1
        )  # list of tensors of len = n_mod, each tensor is of shape batch_size x mod_input_dim
        masks = [x.sum(dim=1) > 0 for x in xs]  # [batch_size] * num_modalities

        recon_loss, modality_recon_losses = self._calc_recon_loss(
            xs, rs, self.losses, integrate_on, size_factor, self.loss_coefs, masks
        )
        kl_loss = kl_weight * kl(Normal(mu, torch.sqrt(torch.exp(logvar))), Normal(0, 1)).sum(dim=1)

        integ_loss = torch.tensor(0.0).to(self.device)
        if self.loss_coefs["integ"] > 0:
            if self.alignment_type == "latent" or self.alignment_type == "both":
                integ_loss += self._calc_integ_loss(z_joint, mu, logvar, integrate_on).to(self.device)
            if self.alignment_type == "marginal" or self.alignment_type == "both":
                for i in range(len(masks)):
                    for j in range(i + 1, len(masks)):
                        idx_where_to_calc_integ_loss = torch.eq(
                            masks[i] == masks[j],
                            torch.eq(masks[i], torch.ones_like(masks[i])),
                        )
                        if (
                            idx_where_to_calc_integ_loss.any()
                        ):  # if need to calc integ loss for a group between modalities
                            marginal_i = z_marginal[:, i, :][idx_where_to_calc_integ_loss]
                            marginal_j = z_marginal[:, j, :][idx_where_to_calc_integ_loss]

                            mu_i = mu_marginal[:, i, :][idx_where_to_calc_integ_loss]
                            mu_j = mu_marginal[:, j, :][idx_where_to_calc_integ_loss]

                            logvar_i = logvar_marginal[:, i, :][idx_where_to_calc_integ_loss]
                            logvar_j = logvar_marginal[:, j, :][idx_where_to_calc_integ_loss]

                            marginals = torch.cat([marginal_i, marginal_j])
                            mus_marginal = torch.cat([mu_i, mu_j])
                            logvars_marginal = torch.cat([logvar_i, logvar_j])

                            modalities = torch.cat(
                                [
                                    torch.Tensor([i] * marginal_i.shape[0]),
                                    torch.Tensor([j] * marginal_j.shape[0]),
                                ]
                            ).to(self.device)

                            integ_loss += self._calc_integ_loss(
                                marginals, mus_marginal, logvars_marginal, modalities
                            ).to(self.device)

                for i in range(len(masks)):
                    marginal_i = z_marginal[:, i, :]
                    marginal_i = marginal_i[masks[i]]

                    mu_i = mu_marginal[:, i, :]
                    mu_i = mu_i[masks[i]]

                    logvar_i = logvar_marginal[:, i, :]
                    logvar_i = logvar_i[masks[i]]

                    group_marginal = integrate_on[masks[i]]
                    integ_loss += self._calc_integ_loss(marginal_i, mu_i, logvar_i, group_marginal).to(self.device)

        loss = torch.mean(
            self.loss_coefs["recon"] * recon_loss
            + self.loss_coefs["kl"] * kl_loss
            + self.loss_coefs["integ"] * integ_loss
        )

        modality_recon_losses = {
            f"modality_{i}_reconstruction_loss": modality_recon_losses[i] for i in range(len(modality_recon_losses))
        }
        extra_metrics = {"integ_loss": integ_loss}
        extra_metrics.update(modality_recon_losses)

        return loss, recon_loss, kl_loss, extra_metrics

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ) -> tuple[
        torch.FloatTensor,
        dict[str, torch.FloatTensor],
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        dict[str, torch.FloatTensor],
    ]:
        """Calculate the (modality) reconstruction loss, Kullback divergences and integration loss.

        Parameters
        ----------
        tensors
            Tensor of values with shape ``(batch_size, n_input_features)``.
        inference_outputs
            Dictionary with the inference output.
        generative_outputs
            Dictionary with the generative output.
        kl_weight
            Weight of the KL loss. Default is 1.0.

        Returns
        -------
        Reconstruction loss, Kullback divergences, integration loss and modality reconstruction losses.
        """
        loss, recon_loss, kl_loss, extra_metrics = self._calculate_loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_loss,
            extra_metrics=extra_metrics,
        )

    # @torch.inference_mode()
    # def sample(self, tensors, n_samples=1):
    #     """Sample from the generative model.

    #     Parameters
    #     ----------
    #     tensors
    #         Tensor of values.
    #     n_samples
    #         Number of samples to generate.

    #     Returns
    #     -------
    #     Generative outputs.
    #     """
    #     inference_kwargs = {"n_samples": n_samples}
    #     with torch.inference_mode():
    #         (
    #             _,
    #             generative_outputs,
    #         ) = self.forward(
    #             tensors,
    #             inference_kwargs=inference_kwargs,
    #             compute_loss=False,
    #         )
    #     return generative_outputs["rs"]

    def _calc_recon_loss(self, xs, rs, losses, group, size_factor, loss_coefs, masks):
        loss = []
        for i, (x, r, loss_type) in enumerate(zip(xs, rs, losses, strict=False)):
            if len(r) != 2 and len(r.shape) == 3:
                r = r.squeeze()
            if loss_type == "mse":
                mse_loss = loss_coefs[str(i)] * torch.sum(nn.MSELoss(reduction="none")(r, x), dim=-1)
                loss.append(mse_loss)
            elif loss_type == "nb":
                dec_mean = r
                size_factor_view = size_factor.expand(dec_mean.size(0), dec_mean.size(1))
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[group.squeeze().long()]
                dispersion = torch.exp(dispersion)
                nb_loss = torch.sum(NegativeBinomial(mu=dec_mean, theta=dispersion).log_prob(x), dim=-1)
                nb_loss = loss_coefs[str(i)] * nb_loss
                loss.append(-nb_loss)
            elif loss_type == "zinb":
                dec_mean, dec_dropout = r
                dec_mean = dec_mean.squeeze()
                dec_dropout = dec_dropout.squeeze()
                size_factor_view = size_factor.unsqueeze(1).expand(dec_mean.size(0), dec_mean.size(1))
                dec_mean = dec_mean * size_factor_view
                dispersion = self.theta.T[group.squeeze().long()]
                dispersion = torch.exp(dispersion)
                zinb_loss = torch.sum(
                    ZeroInflatedNegativeBinomial(mu=dec_mean, theta=dispersion, zi_logits=dec_dropout).log_prob(x),
                    dim=-1,
                )
                zinb_loss = loss_coefs[str(i)] * zinb_loss
                loss.append(-zinb_loss)
            elif loss_type == "bce":
                bce_loss = loss_coefs[str(i)] * torch.sum(torch.nn.BCELoss(reduction="none")(r, x), dim=-1)
                loss.append(bce_loss)

        return (
            torch.sum(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=1),
            torch.sum(torch.stack(loss, dim=-1) * torch.stack(masks, dim=-1), dim=0),
        )

    def _calc_integ_loss(self, z, mu, logvar, group):
        loss = torch.tensor(0.0).to(self.device)
        unique = torch.unique(group)

        if len(unique) > 1:
            if self.modality_alignment == "MMD":
                zs = [z[group == i] for i in unique]
                for i in range(len(zs)):
                    for j in range(i + 1, len(zs)):
                        loss += MMD(kernel_type=self.kernel_type)(zs[i], zs[j])

            elif self.modality_alignment == "Jeffreys":
                mus_joint = [mu[group == i] for i in unique]
                logvars_joint = [logvar[group == i] for i in unique]
                for i in range(len(mus_joint)):
                    for j in range(i + 1, len(mus_joint)):
                        if len(mus_joint[i]) == len(mus_joint[j]):
                            loss += Jeffreys()(
                                (mus_joint[i], torch.exp(logvars_joint[i])), (mus_joint[j], torch.exp(logvars_joint[j]))
                            )
                        else:
                            raise ValueError(
                                f"mus_joint[i] and mus_joint[j] have different lengths: {len(mus_joint[i])} != {len(mus_joint[j])}"
                            )

        return loss

    def _compute_cont_cov_embeddings(self, covs):
        """Compute embeddings for continuous covariates.

        Adapted from
        Title: CPA (c) Facebook, Inc.
        Date: 26.01.2022
        Link to the used code:
        https://github.com/facebookresearch/CPA/blob/382ff641c588820a453d801e5d0e5bb56642f282/compert/model.py#L342

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
            return self.cont_covariate_curves(covs) @ self.cont_covariate_embeddings.weight

    def select_losses_to_plot(self):
        """Select losses to plot.

        Returns
        -------
        Loss names.
        """
        loss_names = ["kl_local", "elbo", "reconstruction_loss"]
        for i in range(self.n_modality):
            loss_names.append(f"modality_{i}_reconstruction_loss")
        if self.loss_coefs["integ"] != 0:
            loss_names.append("integ_loss")
        return loss_names
