import torch
from torch import nn
from torch.nn import functional as F

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi import REGISTRY_KEYS

from ..nn import MLP, Aggregator


class MILClassifierTorch(BaseModuleClass):
    def __init__(
        self,
        z_dim=16,
        dropout=0.2,
        normalization="layer",
        sample_idx=None,
        num_classification_classes=[],  # number of classes for each of the classification task
        scoring="gated_attn",
        attn_dim=16,
        n_layers_cell_aggregator=1,
        n_layers_classifier=1,
        n_layers_mlp_attn=1,
        n_layers_regressor=1,
        n_hidden_regressor=16,
        n_hidden_cell_aggregator=16,
        n_hidden_classifier=16,
        n_hidden_mlp_attn=16,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        sample_batch_size=128,
        attention_dropout=True,
        class_idx=[],  # which indices in cat covariates to do classification on, i.e. exclude from inference; this is a torch tensor
        ord_idx=[],  # which indices in cat covariates to do ordinal regression on and also exclude from inference; this is a torch tensor
        reg_idx=[],  # which indices in cont covariates to do regression on and also exclude from inference; this is a torch tensor
        drop_attn=False,
        aggr="attn",
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

        self.class_loss_coef = class_loss_coef
        self.regression_loss_coef = regression_loss_coef
        self.sample_idx = sample_idx
        self.sample_batch_size = sample_batch_size
        self.aggr = aggr
        self.class_weights = class_weights
        self.anneal_class_loss = anneal_class_loss
        self.num_classification_classes = num_classification_classes
        self.class_idx = class_idx
        self.ord_idx = ord_idx
        self.reg_idx = reg_idx

        self.cell_level_aggregator = nn.Sequential(
            MLP(
                z_dim,
                z_dim,
                n_layers=n_layers_cell_aggregator,
                n_hidden=n_hidden_cell_aggregator,
                dropout_rate=dropout,
                activation=self.activation,
                normalization=normalization,
            ),
            Aggregator(
                n_input=z_dim,
                scoring=scoring,
                attn_dim=attn_dim,
                sample_batch_size=sample_batch_size,
                scale=True,
                attention_dropout=attention_dropout,
                drop_attn=drop_attn,
                dropout=dropout,
                n_layers_mlp_attn=n_layers_mlp_attn,
                n_hidden_mlp_attn=n_hidden_mlp_attn,
                activation=self.activation,
            ),
        )

        if len(self.class_idx) > 0:
            self.classifiers = torch.nn.ModuleList()

            # TODO check/remove other aggr types
            # TODO 2 * z_dim is for `both`, remove 
            # classify zs directly
            class_input_dim = z_dim if self.aggr == 'attn' else 2 * z_dim 

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

        if len(self.ord_idx) + len(self.reg_idx) > 0:
            self.regressors = torch.nn.ModuleList()
            for _ in range(
                len(self.ord_idx) + len(self.reg_idx)
            ):  # one head per standard regression and one per ordinal regression
                if n_layers_regressor == 1:
                    self.regressors.append(nn.Linear(z_dim, 1))
                else:
                    self.regressors.append(
                        nn.Sequential(
                            MLP(
                                z_dim,
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
        return  {"x": x}

    def _get_generative_input(self, tensors, inference_outputs):
        z_joint = inference_outputs["z_joint"]
        return {"z_joint": z_joint}

    @auto_move_data
    def inference(self, x):
        z_joint = x
        inference_outputs = {"z_joint": z_joint}

        # MIL part
        batch_size = x.shape[0]

        idx = list(
            range(self.sample_batch_size, batch_size, self.sample_batch_size)
        )
        if (
            batch_size % self.sample_batch_size != 0
        ):  # can only happen during inference for last batches for each sample
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

        if len(self.class_idx) > 0:
            predictions.extend([classifier(zs_aggr) for classifier in self.classifiers])
        if len(self.ord_idx) + len(self.reg_idx) > 0:
            predictions.extend([regressor(zs_aggr) for regressor in self.regressors])

        inference_outputs.update(
            {"predictions": predictions}
        )  # predictions are a list as they can have different number of classes
        return inference_outputs  # z_joint, mu, logvar, predictions

    @auto_move_data
    def generative(self, z_joint):
        return z_joint

    def _calculate_loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        x = tensors[REGISTRY_KEYS.X_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        # MIL classification loss
        batch_size = x.shape[0]
        # keep indices of the start positions of samples in the batch (but not the first)
        idx = list(
            range(self.sample_batch_size, batch_size, self.sample_batch_size)
        ) 
        if (
            batch_size % self.sample_batch_size != 0
        ):  # can only happen during inference for last batches for each sample
            idx = []

        # TODO put these into functions
        if len(self.reg_idx) > 0:
            regression = torch.index_select(cont_covs, 1, self.reg_idx.to(self.device)) # batch_size x number of regression tasks
            # select the first element of each sample in the batch because we predict on sample level
            regression = regression.view(len(idx) + 1, -1, len(self.reg_idx)) # num_samples_in_batch x sample_batch_size x number of regression tasks
            # take the first element of each sample in the batch as the values are the same for the sample_batch
            regression = regression[:, 0, :] # num_samples_in_batch x number of regression tasks
       
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
        
        predictions = inference_outputs[
            "predictions"
        ]  # list, first from classifiers, then from regressors
        
        accuracies = []
        classification_loss = torch.tensor(0.0).to(self.device)
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
                predictions[len(self.class_idx) + i].squeeze(-1), ordinal_regression[:, i]
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
                predictions[len(self.class_idx) + len(self.ord_idx) + i].squeeze(-1),
                regression[:, i],
            )
            
        accuracy = torch.sum(torch.tensor(accuracies)) / len(accuracies)
              
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

        return loss, recon_loss, kl_loss, extra_metrics

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        loss, recon_loss, kl_loss, extra_metrics = self._calculate_loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )
    
        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_loss,
            extra_metrics=extra_metrics,
        )

    def select_losses_to_plot(self):
        loss_names = []
        if self.class_loss_coef != 0 and len(self.class_idx) > 0:
            loss_names.extend(["class_loss", "accuracy"])
        if self.regression_loss_coef != 0 and len(self.reg_idx) > 0:
            loss_names.append("regression_loss")
        if self.regression_loss_coef != 0 and len(self.ord_idx) > 0:
            loss_names.extend(["regression_loss", "accuracy"])
        return loss_names

    @torch.inference_mode()
    def sample(self, tensors, n_samples=1):
        return self.vae.sample(tensors, n_samples)
