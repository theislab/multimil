import torch

from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi import REGISTRY_KEYS

from ..module import MultiVAETorch, MILClassifierTorch


class MultiVAETorch_MIL(BaseModuleClass):
    def __init__(
        self,
        modality_lengths,
        condition_encoders=False,
        condition_decoders=True,
        normalization="layer",
        z_dim=16,
        losses=[],
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
        sample_idx=None,
        num_classification_classes=[],  # number of classes for each of the classification task
        scoring="gated_attn",
        attn_dim=16,
        cat_covariate_dims=[],
        cont_covariate_dims=[],
        cat_covs_idx=[],
        cont_covs_idx=[],
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
        attention_dropout=True,
        class_idx=[],  # which indices in cat covariates to do classification on, i.e. exclude from inference
        ord_idx=[],  # which indices in cat covariates to do ordinal regression on and also exclude from inference
        reg_idx=[],  # which indices in cont covariates to do regression on and also exclude from inference
        drop_attn=False,
        mmd="latent",
        sample_in_vae=True,
        aggr="attn",
        activation='leaky_relu',
        initialization=None,
        class_weights=None,
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
            aggr=aggr,
            class_weights=class_weights,
            anneal_class_loss=anneal_class_loss,
            num_classification_classes=num_classification_classes,
            class_idx=class_idx,
            ord_idx=ord_idx,
            reg_idx=reg_idx,
            sample_idx=sample_idx,
            drop_attn=drop_attn,
            attention_dropout=attention_dropout,
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
    def inference(self, x, cat_covs, cont_covs):
        # VAE part
        inference_outputs = self.vae_module.inference(x, cat_covs, cont_covs)
        z_joint = inference_outputs["z_joint"]
        
        # MIL part
        mil_inference_outputs = self.mil_module.inference(z_joint)
        inference_outputs.update(mil_inference_outputs)
        return inference_outputs  # z_joint, mu, logvar, z_marginal, predictions

    @auto_move_data
    def generative(self, z_joint, cat_covs, cont_covs):
        return self.vae_module.generative(z_joint, cat_covs, cont_covs)

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        loss_vae, recon_loss, kl_loss, extra_metrics = self.vae_module._calculate_loss(tensors, inference_outputs, generative_outputs, kl_weight)
        loss_mil, _, _, extra_metrics_mil = self.mil_module._calculate_loss(tensors, inference_outputs, generative_outputs, kl_weight)
        loss = loss_vae + loss_mil
        extra_metrics.update(extra_metrics_mil)

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_loss,
            extra_metrics=extra_metrics,
        )