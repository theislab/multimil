import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

from multimil.module import MILClassifierTorch, MultiVAETorch


class MultiVAETorch_MIL(BaseModuleClass):
    """MultiMIL's end-to-end multimodal integration and MIL classification modules.

    Parameters
    ----------
    modality_lengths
        Number of features for each modality.
    condition_encoders
        Whether to condition the encoders on the covariates.
    condition_decoders
        Whether to condition the decoders on the covariates.
    normalization
        Normalization to use in the network.
    z_dim
        Dimensionality of the latent space.
    losses
        List of losses to use in the VAE.
    dropout
        Dropout rate.
    cond_dim
        Dimensionality of the covariate embeddings.
    kernel_type
        Type of kernel to use for the MMD loss.
    loss_coefs
        Coefficients for the different losses.
    num_groups
        Number of groups to use for the MMD loss.
    integrate_on_idx
        Indices of the covariates to integrate on.
    n_layers_encoders
        Number of layers in the encoders.
    n_layers_decoders
        Number of layers in the decoders.
    n_hidden_encoders
        Number of hidden units in the encoders.
    n_hidden_decoders
        Number of hidden units in the decoders.
    num_classification_classes
        Number of classes for each of the classification task.
    scoring
        Scoring function to use for the MIL classification.
    attn_dim
        Dimensionality of the hidden attention dimension.
    cat_covariate_dims
        Number of categories for each of the categorical covariates.
    cont_covariate_dims
        Number of categories for each of the continuous covariates. Always 1.
    cat_covs_idx
        Indices of the categorical covariates.
    cont_covs_idx
        Indices of the continuous covariates.
    cont_cov_type
        Type of continuous covariate.
    n_layers_cell_aggregator
        Number of layers in the cell aggregator.
    n_layers_classifier
        Number of layers in the classifier.
    n_layers_mlp_attn
        Number of layers in the attention MLP.
    n_layers_cont_embed
        Number of layers in the continuous embedding calculation.
    n_layers_regressor
        Number of layers in the regressor.
    n_hidden_regressor
        Number of hidden units in the regressor.
    n_hidden_cell_aggregator
        Number of hidden units in the cell aggregator.
    n_hidden_classifier
        Number of hidden units in the classifier.
    n_hidden_mlp_attn
        Number of hidden units in the attention MLP.
    n_hidden_cont_embed
        Number of hidden units in the continuous embedding calculation.
    class_loss_coef
        Coefficient for the classification loss.
    regression_loss_coef
        Coefficient for the regression loss.
    sample_batch_size
        Bag size.
    class_idx
        Which indices in cat covariates to do classification on.
    ord_idx
        Which indices in cat covariates to do ordinal regression on.
    reg_idx
        Which indices in cont covariates to do regression on.
    mmd
        Type of MMD loss to use.
    activation
        Activation function to use.
    initialization
        Initialization method to use.
    anneal_class_loss
        Whether to anneal the classification loss.
    """

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
        loss_coefs=None,
        num_groups=1,
        integrate_on_idx=None,
        n_layers_encoders=None,
        n_layers_decoders=None,
        n_hidden_encoders=None,
        n_hidden_decoders=None,
        num_classification_classes=None,  # number of classes for each of the classification task
        scoring="gated_attn",
        attn_dim=16,
        cat_covariate_dims=None,
        cont_covariate_dims=None,
        cat_covs_idx=None,
        cont_covs_idx=None,
        cont_cov_type="logsigm",
        n_layers_cell_aggregator=1,
        n_layers_classifier=1,
        n_layers_mlp_attn=1,
        n_layers_cont_embed=1,
        n_layers_regressor=1,
        n_hidden_regressor=16,
        n_hidden_cell_aggregator=16,
        n_hidden_classifier=16,
        n_hidden_mlp_attn=16,
        n_hidden_cont_embed=16,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        sample_batch_size=128,
        class_idx=None,  # which indices in cat covariates to do classification on, i.e. exclude from inference
        ord_idx=None,  # which indices in cat covariates to do ordinal regression on and also exclude from inference
        reg_idx=None,  # which indices in cont covariates to do regression on and also exclude from inference
        mmd="latent",
        activation="leaky_relu",
        initialization=None,
        anneal_class_loss=False,
    ):
        super().__init__()

        self.vae_module = MultiVAETorch(
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
            cat_covs_idx=cat_covs_idx,
            cont_covs_idx=cont_covs_idx,
            cont_cov_type=cont_cov_type,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_cont_embed=n_hidden_cont_embed,
            mmd=mmd,
            activation=activation,
            initialization=initialization,
        )
        self.mil_module = MILClassifierTorch(
            z_dim=z_dim,
            dropout=dropout,
            n_layers_cell_aggregator=n_layers_cell_aggregator,
            n_layers_classifier=n_layers_classifier,
            n_layers_mlp_attn=n_layers_mlp_attn,
            n_layers_regressor=n_layers_regressor,
            n_hidden_regressor=n_hidden_regressor,
            n_hidden_cell_aggregator=n_hidden_cell_aggregator,
            n_hidden_classifier=n_hidden_classifier,
            n_hidden_mlp_attn=n_hidden_mlp_attn,
            class_loss_coef=class_loss_coef,
            regression_loss_coef=regression_loss_coef,
            sample_batch_size=sample_batch_size,
            anneal_class_loss=anneal_class_loss,
            num_classification_classes=num_classification_classes,
            class_idx=class_idx,
            ord_idx=ord_idx,
            reg_idx=reg_idx,
            activation=activation,
            initialization=initialization,
            normalization=normalization,
            scoring=scoring,
            attn_dim=attn_dim,
        )

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
    def inference(self, x, cat_covs, cont_covs) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass for inference.

        Parameters
        ----------
        x
            Input.
        cat_covs
            Categorical covariates to condition on.
        cont_covs
            Continuous covariates to condition on.

        Returns
        -------
        Joint representations, marginal representations, joint mu's and logvar's and predictions.
        """
        # VAE part
        inference_outputs = self.vae_module.inference(x, cat_covs, cont_covs)
        z_joint = inference_outputs["z_joint"]

        # MIL part
        mil_inference_outputs = self.mil_module.inference(z_joint)
        inference_outputs.update(mil_inference_outputs)
        return inference_outputs  # z_joint, mu, logvar, z_marginal, predictions

    @auto_move_data
    def generative(self, z_joint, cat_covs, cont_covs) -> dict[str, torch.Tensor]:
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
        return self.vae_module.generative(z_joint, cat_covs, cont_covs)

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
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
        Reconstruction loss, Kullback divergences, integration loss, modality reconstruction and prediction losses.
        """
        loss_vae, recon_loss, kl_loss, extra_metrics = self.vae_module._calculate_loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )
        loss_mil, _, _, extra_metrics_mil = self.mil_module._calculate_loss(
            tensors, inference_outputs, generative_outputs, kl_weight
        )
        loss = loss_vae + loss_mil
        extra_metrics.update(extra_metrics_mil)

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_loss,
            extra_metrics=extra_metrics,
        )
