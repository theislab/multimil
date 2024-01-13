import torch
from torch import nn
from torch.nn import functional as F

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi import REGISTRY_KEYS

from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from multigrate.module import MultiVAETorch
from multigrate.nn import MLP


class Aggregator(nn.Module):
    def __init__(
        self,
        n_input=None,
        scoring="gated_attn",
        attn_dim=16,  # D
        patient_batch_size=None,
        scale=False,
        attention_dropout=False,
        drop_attn=False,
        dropout=0.2,
        n_layers_mlp_attn=1,
        n_hidden_mlp_attn=16,
        activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.scoring = scoring
        self.patient_batch_size = patient_batch_size
        self.scale = scale

        if self.scoring == "attn":
            self.attn_dim = (
                attn_dim  # attn dim from https://arxiv.org/pdf/1802.04712.pdf
            )
            self.attention = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Tanh(),
                nn.Dropout(dropout) if attention_dropout else nn.Identity(),
                nn.Linear(self.attn_dim, 1, bias=False),
            )
        elif self.scoring == "gated_attn":
            self.attn_dim = attn_dim
            self.attention_V = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Tanh(),
                nn.Dropout(dropout) if attention_dropout else nn.Identity(),
            )

            self.attention_U = nn.Sequential(
                nn.Linear(n_input, self.attn_dim),
                nn.Sigmoid(),
                nn.Dropout(dropout) if attention_dropout else nn.Identity(),
            )

            self.attention_weights = nn.Linear(self.attn_dim, 1, bias=False)

        elif self.scoring == "mlp":

            if n_layers_mlp_attn == 1:
                self.attention = nn.Linear(n_input, 1)
            else:
                self.attention = nn.Sequential(
                    MLP(
                        n_input,
                        n_hidden_mlp_attn,
                        n_layers=n_layers_mlp_attn - 1,
                        n_hidden=n_hidden_mlp_attn,
                        dropout_rate=dropout,
                        activation=activation,
                    ),
                    nn.Linear(n_hidden_mlp_attn, 1),
                )
        self.dropout_attn = nn.Dropout(dropout) if drop_attn else nn.Identity()

    def forward(self, x):
        # if self.scoring == "sum":
        #  return torch.sum(x, dim=0)  # z_dim depricated

        if self.scoring == "attn":
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            self.A = self.attention(x)  # Nx1
            self.A = torch.transpose(self.A, -1, -2)  # 1xN
            self.A = F.softmax(self.A, dim=-1)  # softmax over N

        elif self.scoring == "gated_attn":
            # from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py (accessed 16.09.2021)
            A_V = self.attention_V(x)  # NxD
            A_U = self.attention_U(x)  # NxD
            self.A = self.attention_weights(
                A_V * A_U
            )  # element wise multiplication # Nx1
            self.A = torch.transpose(self.A, -1, -2)  # 1xN
            self.A = F.softmax(self.A, dim=-1)  # softmax over N

        elif self.scoring == "mlp":
            self.A = self.attention(x)  # N
            self.A = torch.transpose(self.A, -1, -2)
            self.A = F.softmax(self.A, dim=-1)

        else:
            raise NotImplementedError(f'scoring = {self.scoring} is not implemented. Has to be one of ["attn", "gated_attn", "mlp"].')

        if self.scale:
            self.A = self.A * self.A.shape[-1] / self.patient_batch_size

        self.A = self.dropout_attn(self.A)

        return torch.bmm(self.A, x).squeeze(dim=1)  # z_dim


class MILClassifierTorch(BaseModuleClass):
    def __init__(
        self,
        modality_lengths,
        condition_encoders=False,
        condition_decoders=True,
        normalization="layer",
        z_dim=16,
        losses=None,
        dropout=0.2,
        cond_dim=16,
        kernel_type="gaussian",
        loss_coefs=[],
        num_groups=1,
        integrate_on_idx=None,
        n_layers_encoders=None,
        n_layers_decoders=None,
        n_hidden_encoders=None,
        n_hidden_decoders=None,
        patient_idx=None,
        num_classification_classes=[],  # number of classes for each of the classification task
        scoring="gated_attn",
        attn_dim=16,
        cat_covariate_dims=[],
        cont_covariate_dims=[],
        cont_cov_type="logsigm",
        n_layers_cell_aggregator=1,
        n_layers_cov_aggregator=1,
        n_layers_classifier=1,
        n_layers_mlp_attn=1,
        n_layers_cont_embed=1,
        n_layers_regressor=1,
        n_hidden_regressor=16,
        n_hidden_cell_aggregator=16,
        n_hidden_cov_aggregator=16,
        n_hidden_classifier=16,
        n_hidden_mlp_attn=16,
        n_hidden_cont_embed=16,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        reg_coef=1.0,
        add_patient_to_classifier=False,
        hierarchical_attn=True,
        patient_batch_size=128,
        regularize_cell_attn=False,
        regularize_cov_attn=False,
        attention_dropout=True,
        class_idx=[],  # which indices in cat covariates to do classification on, i.e. exclude from inference
        ord_idx=[],  # which indices in cat covariates to do ordinal regression on and also exclude from inference
        reg_idx=[],  # which indices in cont covariates to do regression on and also exclude from inference
        drop_attn=False,
        mmd="latent",
        patient_in_vae=True,
        aggr="attn",
        cov_aggr="attn",
        activation='leaky_relu',
        initialization=None,
        class_weights=None,
        anneal_class_loss=False,
    ):
        super().__init__()

        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh
        else:
            raise NotImplementedError(
                f'activation should be one of ["leaky_relu", "tanh"], but activation={activation} was passed.'
            )

        self.vae = MultiVAETorch(
            modality_lengths=modality_lengths,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            z_dim=z_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            num_groups=num_groups,
            integrate_on_idx=integrate_on_idx,
            cat_covariate_dims=cat_covariate_dims,  # only the actual categorical covs are considered here
            cont_covariate_dims=cont_covariate_dims,  # only the actual cont covs are considered here
            cont_cov_type=cont_cov_type,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_cont_embed=n_hidden_cont_embed,
            mmd=mmd,
            activation=activation,
        )

        self.integrate_on_idx = integrate_on_idx
        self.class_loss_coef = class_loss_coef
        self.regression_loss_coef = regression_loss_coef
        self.reg_coef = reg_coef
        self.add_patient_to_classifier = add_patient_to_classifier
        self.patient_idx = patient_idx
        self.hierarchical_attn = hierarchical_attn
        self.patient_batch_size = patient_batch_size
        self.regularize_cell_attn = regularize_cell_attn
        self.regularize_cov_attn = regularize_cov_attn
        self.aggr = aggr
        self.cov_aggr = cov_aggr
        self.class_weights = class_weights
        self.anneal_class_loss = anneal_class_loss
        self.num_classification_classes = num_classification_classes

        self.cat_cov_idx = set(range(len(class_idx) + len(ord_idx) + len(cat_covariate_dims))).difference(set(class_idx)).difference(set(ord_idx))

        if not patient_in_vae:
            self.cat_cov_idx = self.cat_cov_idx - {patient_idx}
        self.cat_cov_idx = torch.tensor(list(self.cat_cov_idx))
        self.cont_cov_idx = torch.tensor(
            list(
                set(range(len(reg_idx) + len(cont_covariate_dims))).difference(
                    set(reg_idx)
                )
            )
        )

        self.class_idx = torch.tensor(class_idx)
        self.ord_idx = torch.tensor(ord_idx)
        self.reg_idx = torch.tensor(reg_idx)

        self.cond_dim = cond_dim
        self.cell_level_aggregator = nn.Sequential(
            MLP(
                z_dim,
                cond_dim,
                n_layers=n_layers_cell_aggregator,
                n_hidden=n_hidden_cell_aggregator,
                dropout_rate=dropout,
                activation=self.activation,
            ),
            Aggregator(
                n_input=cond_dim,
                scoring=scoring,
                attn_dim=attn_dim,
                patient_batch_size=patient_batch_size,
                scale=True,
                attention_dropout=attention_dropout,
                drop_attn=drop_attn,
                dropout=dropout,
                n_layers_mlp_attn=n_layers_mlp_attn,
                n_hidden_mlp_attn=n_hidden_mlp_attn,
                activation=self.activation,
            ),
        )
        if hierarchical_attn and self.cov_aggr in ["attn", "both"]:
            cov_aggr_input_dim = cond_dim

            self.cov_level_aggregator = nn.Sequential(
                MLP(
                    cov_aggr_input_dim,
                    cov_aggr_input_dim,
                    n_layers=n_layers_cov_aggregator,
                    n_hidden=n_hidden_cov_aggregator,
                    dropout_rate=dropout,
                    activation=self.activation,
                ),
                Aggregator(
                    n_input=cov_aggr_input_dim,
                    scoring=scoring,
                    attn_dim=attn_dim,
                    attention_dropout=attention_dropout,
                    drop_attn=drop_attn,
                    dropout=dropout,
                    n_layers_mlp_attn=n_layers_mlp_attn,
                    n_hidden_mlp_attn=n_hidden_mlp_attn,
                    activation=self.activation,
                ),
            )

        self.classifiers = torch.nn.ModuleList()

        if not self.hierarchical_attn: # we classify zs directly
            class_input_dim = cond_dim if self.aggr == 'attn' else 2 * cond_dim
        else: # classify classify aggregated cov info + molecular info
            if self.cov_aggr == 'concat':
                class_input_dim = (len(cat_covariate_dims) + len(cont_covariate_dims) + 1) * cond_dim # 1 for molecular attention
                if self.aggr == 'both':
                    class_input_dim += z_dim
                if not self.add_patient_to_classifier:
                    class_input_dim -= cond_dim
            elif self.cov_aggr == 'both':
                class_input_dim = 2 * cond_dim
            else:
                class_input_dim = cond_dim # attn or average

        if len(self.class_idx) > 0:
            for num in self.num_classification_classes:
                if n_layers_classifier == 1:
                    self.classifiers.append(nn.Linear(class_input_dim, num))
                else:
                    self.classifiers.append(
                        nn.Sequential(
                            MLP(
                                class_input_dim,
                                n_hidden_classifier,
                                n_layers=n_layers_classifier - 1,
                                n_hidden=n_hidden_classifier,
                                dropout_rate=dropout,
                                activation=self.activation,
                            ),
                            nn.Linear(n_hidden_classifier, num),
                        )
                    )

        # TODO if we keep the second head, adjust
        self.regressors = torch.nn.ModuleList()
        for _ in range(
            len(self.ord_idx) + len(self.reg_idx)
        ):  # one head per standard regression and one per ordinal regression
            if n_layers_regressor == 1:
                self.regressors.append(nn.Linear(cond_dim, 1))
            else:
                self.regressors.append(
                    nn.Sequential(
                        MLP(
                            cond_dim,
                            n_hidden_regressor,
                            n_layers=n_layers_regressor - 1,
                            n_hidden=n_hidden_regressor,
                            dropout_rate=dropout,
                            activation=self.activation,
                        ),
                        nn.Linear(n_hidden_regressor, 1),
                    )
                )

        if initialization == 'xavier':
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('leaky_relu'))
        elif initialization == 'kaiming':
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    # following https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138 (accessed 16.08.22)
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in')

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = {"x": x, "cat_covs": cat_covs, "cont_covs": cont_covs}
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z_joint = inference_outputs["z_joint"]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return {"z_joint": z_joint, "cat_covs": cat_covs, "cont_covs": cont_covs}

    @auto_move_data
    def inference(self, x, cat_covs, cont_covs):

        # # VAE part
        # if len(self.cont_cov_idx) > 0:
        #     cont_covs = torch.index_select(
        #         cont_covs, 1, self.cont_cov_idx.to(self.device)
        #     )
        # if len(self.cat_cov_idx) > 0:
        #     cat_covs = torch.index_select(cat_covs, 1, self.cat_cov_idx.to(self.device))

        # inference_outputs = self.vae.inference(x, cat_covs, cont_covs)
        # z_joint = inference_outputs["z_joint"]
        z_joint = x
        inference_outputs = {"z_joint": z_joint}

        # MIL part
        batch_size = x.shape[0]

        idx = list(
            range(self.patient_batch_size, batch_size, self.patient_batch_size)
        )  # or depending on model.train() and model.eval() ???
        if (
            batch_size % self.patient_batch_size != 0
        ):  # can only happen during inference for last batches for each patient
            idx = []
        zs = torch.tensor_split(z_joint, idx, dim=0)
        zs = torch.stack(zs, dim=0) # num of bags x batch_size x z_dim
        zs_attn = self.cell_level_aggregator(zs)  # num of bags x cond_dim
    
        if self.aggr == "both":
            zs = torch.mean(zs, dim=1)
            zs_aggr = torch.cat([zs_attn, zs], dim=-1) # num of bags in batch x (2 * cond_dim) but cond_dim has to be = z_dim #TODO
        else: # "attn"
            zs_aggr = zs_attn
        predictions = []

        if self.hierarchical_attn:
            add_covariate = lambda i: self.add_patient_to_classifier or (
                not self.add_patient_to_classifier and i != self.patient_idx
            )
            cat_embedds = [
                        cat_covariate_embedding(covariate.long())
                        for covariate, cat_covariate_embedding, i in zip(
                            cat_covs.T,
                            self.vae.cat_covariate_embeddings,
                            self.cat_cov_idx,
                        )
                        if add_covariate(i)
                    ]

            if len(cat_embedds) > 0:  
                cat_embedds = torch.stack(
                    cat_embedds,
                    dim=1,
                )
            else:
                cat_embedds = torch.Tensor().to(self.device)  # if the only registered categorical covs are condition and patient, so cat works later
            if self.vae.n_cont_cov > 0:
                if (
                    cont_covs.shape[-1] != self.vae.n_cont_cov
                ):  # shouldn't happen as we got rid of size factors before
                    raise RuntimeError("cont_covs.shape[-1] != self.vae.n_cont_cov")
                cont_embedds = self.vae.compute_cont_cov_embeddings_(cont_covs) # batch_size x num_of_cont_covs x cond_dim
            else:
                cont_embedds = torch.Tensor().to(self.device)

            cov_embedds = torch.cat([cat_embedds, cont_embedds], dim=1) # batch_size x (num_of_cat_covs + num_of_cont_cons) x cond_dim
            cov_embedds = torch.tensor_split(cov_embedds, idx) # tuple of length equal to num of bags in a batch, each element of shape patient_batch_size x num_of_covs x cond_dim
            cov_embedds = torch.stack(cov_embedds, dim=0) # num_of_bags_in_batch x patient_batch_size x num_of_covs x cond_dim 
            # would've used torch.unique here but it's not differentiable
            # second dim is the patient_batch dim so all the values are the same along this dim -> keep only one
            cov_embedds = cov_embedds[:, 0, :, :] # num_of_bags_in_batch x num_of_covs x cond_dim, 

            if self.cov_aggr == 'concat':
                aggr_bag  = torch.split(cov_embedds, 1, dim=1) # tuple of length = num_of_covs, each of shape num_of_bags_in_batch x 1 x cond_dim
                aggr_bag = torch.cat([zs_aggr.unsqueeze(1), *aggr_bag], dim=-1).squeeze(1)
            else: # attn, both or mean, here def only have one head for cell aggr
                aggr_bag_level = torch.cat(
                    [cov_embedds, zs_aggr.unsqueeze(1)], dim=1
                )  # num of bags in batch x num of cat covs + num of cont covs + 1 (molecular information) x cond_dim
                if self.cov_aggr == 'attn':
                    aggr_bag = self.cov_level_aggregator(aggr_bag_level) # final for attn
                if self.cov_aggr == 'both':
                    aggr_bag = self.cov_level_aggregator(aggr_bag_level)
                    average_head = torch.mean(aggr_bag_level, dim=1)
                    aggr_bag = torch.cat([average_head, aggr_bag], dim=-1) # final for both
                if self.cov_aggr == 'mean':
                    aggr_bag = torch.mean(aggr_bag_level, dim=1) # final for mean

            predictions.extend(
                [classifier(aggr_bag) for classifier in self.classifiers]
            )  # each one num of bags in batch x num of classes
            predictions.extend(
                [regressor(aggr_bag) for regressor in self.regressors]
            )
        else: # classify zs aggregated directly
            predictions.extend([classifier(zs_aggr) for classifier in self.classifiers])
            predictions.extend([regressor(zs_aggr) for regressor in self.regressors])

        inference_outputs.update(
            {"predictions": predictions}
        )  # predictions are a list as they can have different number of classes
        return inference_outputs  # z_joint, mu, logvar, predictions

    @auto_move_data
    def generative(self, z_joint, cat_covs, cont_covs):
        # if len(self.cont_cov_idx) > 0:
        #     cont_covs = torch.index_select(
        #         cont_covs, 1, self.cont_cov_idx.to(self.device)
        #     )
        # if len(self.cat_cov_idx) > 0:
        #     cat_covs = torch.index_select(cat_covs, 1, self.cat_cov_idx.to(self.device))
        # return self.vae.generative(z_joint, cat_covs, cont_covs)
        return z_joint

    def orthogonal_regularization(self, weights, axis=0):
        loss = torch.tensor(0.0).to(self.device)
        for weight in weights:
            if axis == 1:
                weight = weight.T
            dim = weight.shape[1]
            loss += torch.sqrt(
                torch.sum(
                    (
                        torch.matmul(weight.T, weight) - torch.eye(dim).to(self.device)
                    ).pow(2)
                )
            )
        return loss

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.integrate_on_idx is not None:
            integrate_on = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)[:, self.integrate_on_idx]
        else:
            integrate_on = torch.zeros(x.shape[0], 1).to(self.device)

        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)

        # MIL classification loss
        batch_size = x.shape[0]
        idx = list(
            range(self.patient_batch_size, batch_size, self.patient_batch_size)
        )  # or depending on model.train() and model.eval() ???
        if (
            batch_size % self.patient_batch_size != 0
        ):  # can only happen during inference for last batches for each patient
            idx = []

        # TODO in a function
        if len(self.reg_idx) > 0:
            regression = torch.index_select(cont_covs, 1, self.reg_idx.to(self.device))
            regression = regression.view(len(idx) + 1, -1, len(self.reg_idx))[:, 0, :]
        if len(self.cont_cov_idx) > 0:
            cont_covs = torch.index_select(
                cont_covs, 1, self.cont_cov_idx.to(self.device)
            )

        if len(self.ord_idx) > 0:
            ordinal_regression = torch.index_select(
                cat_covs, 1, self.ord_idx.to(self.device)
            )
            ordinal_regression = ordinal_regression.view(
                len(idx) + 1, -1, len(self.ord_idx)
            )[:, 0, :]
        if len(self.class_idx) > 0:
            classification = torch.index_select(
                cat_covs, 1, self.class_idx.to(self.device)
            )
            classification = classification.view(len(idx) + 1, -1, len(self.class_idx))[
                :, 0, :
            ]
        if len(self.cat_cov_idx) > 0:
            cat_covs = torch.index_select(cat_covs, 1, self.cat_cov_idx.to(self.device))

        predictions = inference_outputs[
            "predictions"
        ]  # list, first from classifiers, then from regressors
        
        classification_loss = torch.tensor(0.0).to(self.device)
        accuracies = []
        for i in range(len(self.class_idx)):
            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(self.device)
            classification_loss += F.cross_entropy(
                predictions[i], classification[:, i].long(), weight=self.class_weights
            )  # assume same in the batch
            accuracies.append(
                torch.sum(
                    torch.eq(torch.argmax(predictions[i], dim=-1), classification[:, i])
                )
                / classification[:, i].shape[0]
            )
        
        regression_loss = torch.tensor(0.0).to(self.device)
        for i in range(len(self.ord_idx)):
            regression_loss += F.mse_loss(
                predictions[len(self.class_idx) + i].squeeze(), ordinal_regression[:, i]
            )
            accuracies.append(
                torch.sum(
                    torch.eq(
                        torch.clamp(torch.round(predictions[len(self.class_idx) + i].squeeze()), min=0.0, max=self.num_classification_classes[i] - 1.0),
                        ordinal_regression[:, i],
                    )
                )
                / ordinal_regression[:, i].shape[0]
            )
        for i in range(len(self.reg_idx)):
            regression_loss += F.mse_loss(
                predictions[len(self.class_idx) + len(self.ord_idx) + i].squeeze(),
                regression[:, i],
            )
        accuracy = torch.sum(torch.tensor(accuracies)) / len(accuracies)
        
        # what to regularize:
        weights = []
        if self.regularize_cov_attn:
            weights.append(self.cov_level_aggregator[1].attention_U[0].weight)
            weights.append(self.cov_level_aggregator[1].attention_V[0].weight)
        if self.regularize_cell_attn:
            weights.append(self.cell_level_aggregator[1].attention_U[0].weight)
            weights.append(self.cell_level_aggregator[1].attention_V[0].weight)

        class_loss_anneal_coef = kl_weight if self.anneal_class_loss else 1.0

        loss = torch.mean(
            self.class_loss_coef * classification_loss * class_loss_anneal_coef
            + self.regression_loss_coef * regression_loss
        )

        extra_metrics = {
            "class_loss": classification_loss,
            "accuracy": accuracy,
            "regression_loss": regression_loss,
            }

        recon_loss = torch.zeros(batch_size)
        kl_loss = torch.zeros(batch_size)

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_loss,
            extra_metrics=extra_metrics,
        )

    @torch.inference_mode()
    def sample(self, tensors, n_samples=1):
        return self.vae.sample(tensors, n_samples)
