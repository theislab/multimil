import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.nn import functional as F

from multimil.nn import MLP, Aggregator
from multimil.utils import prep_minibatch, select_covariates


class MILClassifierTorch(BaseModuleClass):
    """MultiMIL's MIL classification module.

    Parameters
    ----------
    z_dim
        Latent dimension.
    dropout
        Dropout rate.
    normalization
        Normalization type.
    num_classification_classes
        Number of classes for each of the classification task.
    scoring
        Scoring type. One of ["gated_attn", "attn", "mean", "max", "sum"].
    attn_dim
        Hidden attention dimension.
    n_layers_cell_aggregator
        Number of layers in the cell aggregator.
    n_layers_classifier
        Number of layers in the classifier.
    n_layers_regressor
        Number of layers in the regressor.
    n_hidden_regressor
        Hidden dimension in the regressor.
    n_hidden_cell_aggregator
        Hidden dimension in the cell aggregator.
    n_hidden_classifier
        Hidden dimension in the classifier.
    class_loss_coef
        Classification loss coefficient.
    regression_loss_coef
        Regression loss coefficient.
    sample_batch_size
        Sample batch size.
    class_idx
        Which indices in cat covariates to do classification on.
    ord_idx
        Which indices in cat covariates to do ordinal regression on.
    reg_idx
        Which indices in cont covariates to do regression on.
    activation
        Activation function.
    initialization
        Initialization type.
    anneal_class_loss
        Whether to anneal the classification loss.
    """

    def __init__(
        self,
        z_dim=16,
        dropout=0.2,
        normalization="layer",
        num_classification_classes=None,  # number of classes for each of the classification task
        scoring="gated_attn",
        attn_dim=16,
        n_layers_cell_aggregator=1,
        n_layers_classifier=2,
        n_layers_regressor=2,
        n_hidden_regressor=128,
        n_hidden_cell_aggregator=128,
        n_hidden_classifier=128,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        sample_batch_size=128,
        class_idx=None,  # which indices in cat covariates to do classification on, i.e. exclude from inference; this is a torch tensor
        ord_idx=None,  # which indices in cat covariates to do ordinal regression on and also exclude from inference; this is a torch tensor
        reg_idx=None,  # which indices in cont covariates to do regression on and also exclude from inference; this is a torch tensor
        activation="leaky_relu",
        initialization=None,
        anneal_class_loss=False,
    ):
        super().__init__()

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        else:
            raise NotImplementedError(
                f'activation should be one of ["leaky_relu", "tanh"], but activation={activation} was passed.'
            )

        self.class_loss_coef = class_loss_coef
        self.regression_loss_coef = regression_loss_coef
        self.sample_batch_size = sample_batch_size
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
                dropout=dropout,
                activation=self.activation,
            ),
        )

        if len(self.class_idx) > 0:
            self.classifiers = torch.nn.ModuleList()

            # classify zs directly
            class_input_dim = z_dim

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

        if initialization == "xavier":
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("leaky_relu"))
        elif initialization == "kaiming":
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    # following https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138 (accessed 16.08.22)
                    nn.init.kaiming_normal_(layer.weight, mode="fan_in")

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        return {"x": x}

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        return {"z": z}

    @auto_move_data
    def inference(self, x) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass for inference.

        Parameters
        ----------
        x
            Input.

        Returns
        -------
        Predictions.
        """
        z = x
        inference_outputs = {"z": z}

        # MIL part
        batch_size = x.shape[0]

        idx = list(range(self.sample_batch_size, batch_size, self.sample_batch_size))
        if (
            batch_size % self.sample_batch_size != 0
        ):  # can only happen during inference for last batches for each sample
            idx = []
        zs = torch.tensor_split(z, idx, dim=0)
        zs = torch.stack(zs, dim=0)  # num of bags x batch_size x z_dim
        zs_attn = self.cell_level_aggregator(zs)  # num of bags x cond_dim

        predictions = []
        if len(self.class_idx) > 0:
            predictions.extend([classifier(zs_attn) for classifier in self.classifiers])
        if len(self.ord_idx) + len(self.reg_idx) > 0:
            predictions.extend([regressor(zs_attn) for regressor in self.regressors])

        inference_outputs.update(
            {"predictions": predictions}
        )  # predictions are a list as they can have different number of classes
        return inference_outputs

    @auto_move_data
    def generative(self, z) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass for generative.

        Parameters
        ----------
        z
            Latent embeddings.

        Returns
        -------
        Tensor of same shape as input.
        """
        return {"z": z}

    def _calculate_loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        # MIL classification loss
        minibatch_size, n_samples_in_batch = prep_minibatch(cat_covs, self.sample_batch_size)
        regression = select_covariates(cont_covs, self.reg_idx.to(self.device), n_samples_in_batch)
        ordinal_regression = select_covariates(cat_covs, self.ord_idx.to(self.device), n_samples_in_batch)
        classification = select_covariates(cat_covs, self.class_idx.to(self.device), n_samples_in_batch)

        predictions = inference_outputs["predictions"]  # list, first from classifiers, then from regressors

        accuracies = []
        classification_loss = torch.tensor(0.0).to(self.device)
        for i in range(len(self.class_idx)):
            classification_loss += F.cross_entropy(
                predictions[i], classification[:, i].long()
            )  # assume same in the batch
            accuracies.append(
                torch.sum(torch.eq(torch.argmax(predictions[i], dim=-1), classification[:, i]))
                / classification[:, i].shape[0]
            )

        regression_loss = torch.tensor(0.0).to(self.device)
        for i in range(len(self.ord_idx)):
            regression_loss += F.mse_loss(predictions[len(self.class_idx) + i].squeeze(-1), ordinal_regression[:, i])
            accuracies.append(
                torch.sum(
                    torch.eq(
                        torch.clamp(
                            torch.round(predictions[len(self.class_idx) + i].squeeze()),
                            min=0.0,
                            max=self.num_classification_classes[i] - 1.0,
                        ),
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

        class_loss_anneal_coef = kl_weight if self.anneal_class_loss else 1.0

        loss = torch.mean(
            self.class_loss_coef * classification_loss * class_loss_anneal_coef
            + self.regression_loss_coef * regression_loss
        )

        extra_metrics = {
            "class_loss": classification_loss,
            "regression_loss": regression_loss,
        }

        if len(accuracies) > 0:
            accuracy = torch.sum(torch.tensor(accuracies)) / len(accuracies)
            extra_metrics["accuracy"] = accuracy

        # don't need in this model but have to return
        recon_loss = torch.zeros(minibatch_size)
        kl_loss = torch.zeros(minibatch_size)

        return loss, recon_loss, kl_loss, extra_metrics

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Loss calculation.

        Parameters
        ----------
        tensors
            Input tensors.
        inference_outputs
            Inference outputs.
        generative_outputs
            Generative outputs.
        kl_weight
            KL weight. Default is 1.0.

        Returns
        -------
        Prediction loss.
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

    def select_losses_to_plot(self):
        """Select losses to plot.

        Returns
        -------
        Loss names.
        """
        loss_names = []
        if self.class_loss_coef != 0 and len(self.class_idx) > 0:
            loss_names.extend(["class_loss", "accuracy"])
        if self.regression_loss_coef != 0 and len(self.reg_idx) > 0:
            loss_names.append("regression_loss")
        if self.regression_loss_coef != 0 and len(self.ord_idx) > 0:
            loss_names.extend(["regression_loss", "accuracy"])
        return loss_names
