import torch
import scipy
import logging
import pandas as pd
import numpy as np
import anndata as ad
from matplotlib import pyplot as plt
import warnings

from ..model import MultiVAE, MILClassifier
from ..dataloaders import GroupDataSplitter, GroupAnnDataLoader
from ..module import MultiVAETorch_MIL
from ..utils import create_df
from typing import List, Optional, Union, Dict
from math import ceil
from scvi.model.base import BaseModelClass, ArchesMixin
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.train._callbacks import SaveBestState
from pytorch_lightning.callbacks import ModelCheckpoint
from scvi.train import TrainRunner, AdversarialTrainingPlan
from scvi import REGISTRY_KEYS
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.data import AnnDataManager, fields
from anndata import AnnData
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base._utils import _initialize_model

logger = logging.getLogger(__name__)


class MultiVAE_MIL(BaseModelClass, ArchesMixin):
    def __init__(
        self,
        adata,
        sample_key,
        classification=[],
        regression=[],
        ordinal_regression=[],
        sample_batch_size=128,
        integrate_on=None,
        condition_encoders=False,
        condition_decoders=True,
        normalization="layer",
        n_layers_encoders=None,
        n_layers_decoders=None,
        n_hidden_encoders=None,
        n_hidden_decoders=None,
        z_dim=30,
        losses=[],
        dropout=0.2,
        cond_dim=16,
        kernel_type="gaussian",
        loss_coefs=[],
        scoring="gated_attn",
        attn_dim=16,
        n_layers_cell_aggregator: int = 1,
        n_layers_classifier: int = 2,
        n_layers_regressor: int = 1,
        n_layers_mlp_attn: int = 1,
        n_layers_cont_embed: int = 1,
        n_hidden_cell_aggregator: int = 128,
        n_hidden_classifier: int = 128,
        n_hidden_cont_embed: int = 128,
        n_hidden_mlp_attn: int = 32,
        n_hidden_regressor: int = 128,
        attention_dropout=False,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        cont_cov_type="logsigm",
        drop_attn=False,
        mmd="latent",
        sample_in_vae=True,
        aggr="attn", # or 'both' = attn + average (two heads)
        activation='leaky_relu', # or tanh
        initialization='kaiming', # xavier (tanh) or kaiming (leaky_relu)
        weighted_class_loss=False, 
        anneal_class_loss=False,
    ):
        super().__init__(adata)
        # TODO figure out what's happening with sample_in_vae

        # add prediction covariates to ignore_covariates_vae
        ignore_covariates_vae = []
        if sample_in_vae is False:
            ignore_covariates_vae.append(sample_key)
        for key in classification + ordinal_regression + regression:
                ignore_covariates_vae.append(key)

        setup_args = self.adata_manager.registry["setup_args"]
        setup_args.pop('ordinal_regression_order')
        MultiVAE.setup_anndata(
            adata,
            **setup_args,
        )
        # TODO check if need/can set self.multivae.adata = None

        self.multivae = MultiVAE(
            adata=adata,
            integrate_on=integrate_on,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            z_dim=z_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            cont_cov_type=cont_cov_type,
            n_layers_cont_embed=n_layers_cont_embed,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_hidden_cont_embed=n_hidden_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            mmd=mmd,
            activation=activation,
            initialization=initialization,
            ignore_covariates=ignore_covariates_vae,
        )

        # add all actual categorical covariates to ignore_covariates_mil
        ignore_covariates_mil = []
        if self.adata_manager.registry["setup_args"]["categorical_covariate_keys"] is not None:
            for cat_name in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
                if cat_name not in classification + ordinal_regression + [sample_key]:
                    ignore_covariates_mil.append(cat_name)
        # add all actual continuous covariates to ignore_covariates_mil
        if self.adata_manager.registry["setup_args"]["continuous_covariate_keys"] is not None:
            for cat_name in self.adata_manager.registry["setup_args"]["continuous_covariate_keys"]:
                if cat_name not in regression:
                    ignore_covariates_mil.append(cat_name)

        setup_args = self.adata_manager.registry["setup_args"]
        setup_args.pop('batch_key')
        setup_args.pop('size_factor_key')
        setup_args.pop('rna_indices_end')

        MILClassifier.setup_anndata(
            adata=adata,
            **setup_args,
        )

        self.mil = MILClassifier(
            adata=adata,
            sample_key=sample_key,
            classification=classification,
            regression=regression,
            ordinal_regression=ordinal_regression,
            sample_batch_size=sample_batch_size,
            normalization=normalization,
            z_dim=z_dim,
            dropout=dropout,
            scoring=scoring,
            attn_dim=attn_dim,
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
            attention_dropout=attention_dropout,
            drop_attn=drop_attn,
            aggr=aggr,
            activation=activation,
            initialization=initialization,
            weighted_class_loss=weighted_class_loss,
            anneal_class_loss=anneal_class_loss,
            ignore_covariates=ignore_covariates_mil,
        )

        # clear up the memory
        self.multivae.module = None
        self.mil.module = None

        self.sample_in_vae = sample_in_vae
       
        self.module = MultiVAETorch_MIL(
            # vae
            modality_lengths=self.multivae.modality_lengths,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            activation=activation,
            initialization=initialization,
            z_dim=z_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            num_groups=self.multivae.num_groups,
            integrate_on_idx=self.multivae.integrate_on_idx,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_cont_embed=n_hidden_cont_embed,
            cat_covariate_dims=self.multivae.cat_covariate_dims,
            cont_covariate_dims=self.multivae.cont_covariate_dims,
            cat_cov_idx=self.multivae.cat_covs_idx,
            cont_cov_idx=self.multivae.cont_covs_idx,
            cont_cov_type=cont_cov_type,
            mmd=mmd,
            # mil
            num_classification_classes=self.mil.num_classification_classes,
            scoring=scoring,
            attn_dim=attn_dim,
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
            sample_idx=self.mil.sample_idx, # TODO I think we don't need it any more
            sample_batch_size=sample_batch_size,
            attention_dropout=attention_dropout,
            class_idx=self.mil.class_idx,
            ord_idx=self.mil.ord_idx,
            reg_idx=self.mil.regression_idx,
            drop_attn=drop_attn,
            sample_in_vae=sample_in_vae,
            aggr=aggr,
            class_weights=self.mil.class_weights,
            anneal_class_loss=anneal_class_loss,
        )

        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 200,
        lr: float = 5e-4,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        adversarial_mixing: bool = False,
        plan_kwargs: Optional[dict] = None,
        save_checkpoint_every_n_epochs: Optional[int] = None,
        path_to_checkpoints: Optional[str] = None,
        early_stopping_monitor: Optional[str] = "accuracy_validation",
        early_stopping_mode: Optional[str] = "max",
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            Save the best model state with respect to the validation loss, or use the final
            state in the training procedure
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
            If so, val is checked every epoch.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if len(self.mil.regression) > 0:
            if early_stopping_monitor == "accuracy_validation":
                warnings.warn("Setting early_stopping_monitor to 'regression_loss_validation' and early_stopping_mode to 'min' as regression is used.")
                early_stopping_monitor = "regression_loss_validation"
                early_stopping_mode = "min"
        if n_epochs_kl_warmup is None:
            n_epochs_kl_warmup = max(max_epochs // 3, 1)
        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_mixing,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "scale_adversarial_loss": 1,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if save_best:
            if "callbacks" not in kwargs.keys():
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(SaveBestState(monitor=early_stopping_monitor,  mode=early_stopping_mode))

        if save_checkpoint_every_n_epochs is not None:
            if path_to_checkpoints is not None:
                kwargs["callbacks"].append(ModelCheckpoint(
                    dirpath = path_to_checkpoints,
                    save_top_k = -1,
                    monitor = 'epoch',
                    every_n_epochs = save_checkpoint_every_n_epochs,
                    verbose = True,
                ))
            else:
                raise ValueError(f"`save_checkpoint_every_n_epochs` = {save_checkpoint_every_n_epochs} so `path_to_checkpoints` has to be not None but is {path_to_checkpoints}.")

        data_splitter = GroupDataSplitter(
            self.adata_manager,
            group_column=self.mil.sample_key,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = AdversarialTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_mode=early_stopping_mode,
            early_stopping_patience=50,
            enable_checkpointing=True,
            **kwargs,
        )
        return runner()

    @classmethod
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        batch_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        rna_indices_end: Optional[int] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        ordinal_regression_order: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        """Set up :class:`~anndata.AnnData` object.

        A mapping will be created between data fields used by ``scvi`` to their respective locations in adata.
        This method will also compute the log mean and log variance per batch for the library size prior.
        None of the data in adata are modified. Only adds fields to adata.

        :param adata:
            AnnData object containing raw counts. Rows represent cells, columns represent features
        :param rna_indices_end:
            Integer to indicate where RNA feature end in the AnnData object. May be needed to calculate ``libary_size``.
        :param categorical_covariate_keys:
            Keys in `adata.obs` that correspond to categorical data
        :param continuous_covariate_keys:
            Keys in `adata.obs` that correspond to continuous data
        :param ordinal_regression_order:
            Dictionary with regression classes as keys and order of classes as values
        """

        # TODO duplicate code here and in _multivae.py, move to function
        if ordinal_regression_order is not None:
            if not set(ordinal_regression_order.keys()).issubset(
                categorical_covariate_keys
            ):
                raise ValueError(
                    f"All keys {ordinal_regression_order.keys()} has to be registered as categorical covariates too, but categorical_covariate_keys = {categorical_covariate_keys}"
                )
            for key in ordinal_regression_order.keys():
                adata.obs[key] = adata.obs[key].astype("category")
                if set(adata.obs[key].cat.categories) != set(
                    ordinal_regression_order[key]
                ):
                    raise ValueError(
                        f"Categories of adata.obs[{key}]={adata.obs[key].cat.categories} are not the same as categories specified = {ordinal_regression_order[key]}"
                    )
                adata.obs[key] = adata.obs[key].cat.reorder_categories(
                    ordinal_regression_order[key]
                )

        if size_factor_key is not None and rna_indices_end is not None:
            raise ValueError("Only one of [`size_factor_key`, `rna_indices_end`] can be specified, but both are not `None`.")
        # TODO change to when both are None, use all input features to calculate the size factors, add warning 
        if size_factor_key is None and rna_indices_end is None:
            raise ValueError("One of [`size_factor_key`, `rna_indices_end`] has to be specified, but both are `None`.")

        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer=None,),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            fields.NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]

        # only one can be not None
        if size_factor_key is not None:
            anndata_fields.append(fields.NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key
            ))

        if rna_indices_end is not None:
            if scipy.sparse.issparse(adata.X):
                adata.obs.loc[:, "size_factors"] = adata[:, :rna_indices_end].X.A.sum(1).T.tolist()
            else:
                adata.obs.loc[:, "size_factors"] = adata[:, :rna_indices_end].X.sum(1).T.tolist()
            anndata_fields.append(fields.NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, "size_factors"
            ))
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_model_output(
        self,
        adata=None,
        batch_size=256,
    ):
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata,
            batch_size=batch_size,
            min_size_per_class=batch_size,  # hack to ensure that not full batches are processed properly
            data_loader_class=GroupAnnDataLoader,
            shuffle=False,
            shuffle_classes=False,
            group_column=self.mil.sample_key,
            drop_last=False,
        )

        # TODO duplicate code with _mil.py, move to function
        latent, cell_level_attn = [], []
        class_pred, ord_pred, reg_pred = {}, {}, {}
        (
            bag_class_true,
            bag_class_pred,
            bag_reg_true,
            bag_reg_pred,
            bag_ord_true,
            bag_ord_pred,
        ) = ({}, {}, {}, {}, {}, {})

        for tensors in scdl:

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            batch_size = cat_covs.shape[0]

            idx = list(
                range(
                    self.module.mil_module.sample_batch_size,
                    batch_size,
                    self.module.mil_module.sample_batch_size,
                )
            )
            if (
                batch_size % self.module.mil_module.sample_batch_size != 0
            ):  # can only happen during inference for last batches for each sample
                idx = []

            if len(self.mil.regression_idx) > 0:
                regression = torch.index_select(cont_covs, 1, self.mil.regression_idx)
                regression = regression.view(
                    len(idx) + 1, -1, len(self.mil.regression_idx)
                )[:, 0, :]

            if len(self.mil.ord_idx) > 0:
                ordinal_regression = torch.index_select(cat_covs, 1, self.mil.ord_idx)
                ordinal_regression = ordinal_regression.view(
                    len(idx) + 1, -1, len(self.mil.ord_idx)
                )[:, 0, :]
            if len(self.mil.class_idx) > 0:
                classification = torch.index_select(cat_covs, 1, self.mil.class_idx)
                classification = classification.view(
                    len(idx) + 1, -1, len(self.mil.class_idx)
                )[:, 0, :]

            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z_joint"]
            pred = outputs["predictions"]
            cell_attn = self.module.mil_module.cell_level_aggregator[-1].A.squeeze(dim=1)
            size = cell_attn.shape[-1]
            cell_attn = (
                cell_attn.flatten()
            )  # in inference always one patient per batch

            for i in range(len(self.mil.class_idx)):
                bag_class_pred[i] = bag_class_pred.get(i, []) + [pred[i].cpu()]
                bag_class_true[i] = bag_class_true.get(i, []) + [
                    classification[:, i].cpu()
                ]
                class_pred[i] = class_pred.get(i, []) + [
                    pred[i].unsqueeze(1).repeat(1, size, 1).flatten(0, 1)
                ]
            for i in range(len(self.mil.ord_idx)):
                bag_ord_pred[i] = bag_ord_pred.get(i, []) + [
                    pred[len(self.mil.class_idx) + i]
                ]
                bag_ord_true[i] = bag_ord_true.get(i, []) + [
                    ordinal_regression[:, i].cpu()
                ]
                ord_pred[i] = ord_pred.get(i, []) + [
                    pred[len(self.mil.class_idx) + i].repeat(1, size).flatten()
                ]
            for i in range(len(self.mil.regression_idx)):
                bag_reg_pred[i] = bag_reg_pred.get(i, []) + [
                    pred[len(self.mil.class_idx) + len(self.mil.ord_idx) + i].cpu()
                ]
                bag_reg_true[i] = bag_reg_true.get(i, []) + [regression[:, i].cpu()]
                reg_pred[i] = reg_pred.get(i, []) + [
                    pred[len(self.mil.class_idx) + len(self.mil.ord_idx) + i]
                    .repeat(1, size)
                    .flatten()
                ]

            latent += [z.cpu()]
            cell_level_attn += [cell_attn.cpu()]

        latent = torch.cat(latent).numpy()
        cell_level = torch.cat(cell_level_attn).numpy()

        adata.obsm["X_multiMIL"] = latent
        adata.obs["cell_attn"] = cell_level

        for i in range(len(self.mil.class_idx)):
            name = self.mil.classification[i]
            classes = self.adata_manager.get_state_registry('extra_categorical_covs')['mappings'][name]
            df = create_df(class_pred[i], classes, index=adata.obs_names)
            adata.obsm[f"classification_predictions_{name}"] = df
            adata.obs[f"predicted_{name}"] = df.to_numpy().argmax(axis=1)
            adata.obs[f"predicted_{name}"] = adata.obs[f"predicted_{name}"].astype(
                "category"
            )
            adata.obs[f"predicted_{name}"] = adata.obs[
                f"predicted_{name}"
            ].cat.rename_categories({i: cl for i, cl in enumerate(classes)})
            adata.uns[f"bag_classification_true_{name}"] = create_df(
                bag_class_true, self.mil.classification
            )
            df_bag = create_df(bag_class_pred[i], classes)
            adata.uns[f"bag_classification_predictions_{name}"] = df_bag
            adata.uns[f"bag_predicted_{name}"] = df_bag.to_numpy().argmax(axis=1)
        for i in range(len(self.mil.ord_idx)):
            name = self.mil.ordinal_regression[i]
            classes = self.adata_manager.get_state_registry('extra_categorical_covs')['mappings'][name]
            df = create_df(ord_pred[i], columns=['pred_regression_value'], index=adata.obs_names)
            adata.obsm[f"ord_regression_predictions_{name}"] = df
            adata.obs[f"predicted_{name}"] = np.clip(np.round(df.to_numpy()), a_min=0.0, a_max=len(classes) - 1.0)
            adata.obs[f"predicted_{name}"] = adata.obs[f"predicted_{name}"].astype(int).astype("category")
            adata.obs[f"predicted_{name}"] = adata.obs[
                f"predicted_{name}"
            ].cat.rename_categories({i: cl for i, cl in enumerate(classes)})
            adata.uns[f"bag_ord_regression_true_{name}"] = create_df(bag_ord_true, self.mil.ordinal_regression)
            df_bag = create_df(bag_ord_pred[i], ['predicted_regression_value'])
            adata.uns[f"bag_ord_regression_predictions_{name}"] = df_bag
            adata.uns[f"bag_predicted_{name}"] = np.clip(np.round(df_bag.to_numpy()), a_min=0.0, a_max=len(classes) - 1.0)
        if len(self.mil.regression_idx) > 0:
            adata.obsm["regression_predictions"] = create_df(
                reg_pred, self.mil.regression, index=adata.obs_names
            )
            adata.uns["bag_regression_true"] = create_df(
                bag_reg_true, self.mil.regression
            )
            adata.uns["bag_regression_predictions"] = create_df(
                bag_reg_pred, self.mil.regression
            )

    def plot_losses(self, save=None):
        """Plot losses."""
        df = pd.DataFrame(self.history["train_loss_epoch"])
        for key in self.history.keys():
            if key != "train_loss_epoch":
                df = df.join(self.history[key])

        df["epoch"] = df.index

        loss_names = ["kl_local", "elbo", "reconstruction_loss"]
        for i in range(self.module.vae_module.n_modality):
            loss_names.append(f'modality_{i}_reconstruction_loss')

        if self.module.vae_module.loss_coefs["integ"] != 0:
            loss_names.append("integ_loss")

        # TODO check if better to get .class_idx etc from model or module
        if self.module.mil_module.class_loss_coef != 0 and len(self.module.mil_module.class_idx) > 0:
            loss_names.extend(["class_loss", "accuracy"])
        
        if self.module.mil_module.regression_loss_coef != 0 and len(self.module.mil_module.reg_idx) > 0:
            loss_names.append("regression_loss")
        
        if self.module.mil_module.regression_loss_coef != 0 and len(self.module.mil_module.ord_idx) > 0:
            loss_names.extend(["regression_loss", "accuracy"])

        nrows = ceil(len(loss_names) / 2)

        plt.figure(figsize=(15, 5 * nrows))

        for i, name in enumerate(loss_names):
            plt.subplot(nrows, 2, i + 1)
            plt.plot(df["epoch"], df[name + "_train"], ".-", label=name + "_train")
            plt.plot(df["epoch"], df[name + "_validation"], ".-", label=name + "_validation")
            plt.xlabel("epoch")
            plt.legend()
        if save is not None:
            plt.savefig(save, bbox_inches="tight")


    # not updated yet TODO
    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/model/base/_archesmixin.py#L30
    # accessed on 7 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        # use_prediction_labels: bool,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
        freeze: bool = True,
        ignore_covariates: Optional[List[str]] = None,
    ):
        """Online update of a reference model with scArches algorithm # TODO cite.

        :param adata:
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        :param use_prediction_labels:
            Whether to use prediction labels to fine-tune both parts of the model 
            or not, in which case only the VAE gets fine-tuned.
        :param reference_model:
            Already instantiated model of the same class
        :param use_gpu:
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False)
        :param freeze:
            Whether to freeze the encoders and decoders and only train the new weights
        """
        _, _, device = parse_use_gpu_arg(use_gpu)

        attr_dict, var_names, load_state_dict = _get_loaded_data(
            reference_model, device=device
        )

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError(
                "It appears you are loading a model from a different class."
            )

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError(
                "Saved model does not contain original setup inputs. "
                "Cannot load the original setup."
            )

        # TODO need to make sure that the og registry is saved and not the one after the mil model init
        print(registry[_SETUP_ARGS_KEY])

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)

        # adata_manager = model.get_anndata_manager(adata, required=True)

        ignore_covariates = []
        if not reference_model.patient_in_vae:
            ignore_covariates = [reference_model.patient_column]

        ref_adata = reference_model.adata.copy()
        setup_args = reference_model.adata_manager.registry["setup_args"].copy()
        setup_args.pop('ordinal_regression_order')

        MultiVAE.setup_anndata(ref_adata, **setup_args)

        vae = MultiVAE(
            ref_adata,
            losses=reference_model.module.vae.losses,
            loss_coefs=reference_model.module.vae.loss_coefs,
            integrate_on=reference_model.module.vae.integrate_on_idx,
            condition_encoders=reference_model.module.vae.condition_encoders,
            condition_decoders=reference_model.module.vae.condition_decoders,
            normalization=reference_model.module.vae.normalization,
            z_dim=reference_model.module.vae.z_dim,
            dropout=reference_model.module.vae.dropout,
            cond_dim=reference_model.module.vae.cond_dim,
            kernel_type=reference_model.module.vae.kernel_type,
            n_layers_encoders=reference_model.module.vae.n_layers_encoders,
            n_layers_decoders=reference_model.module.vae.n_layers_decoders,
            n_hidden_encoders=reference_model.module.vae.n_hidden_encoders,
            n_hidden_decoders=reference_model.module.vae.n_hidden_decoders,
            cont_cov_type=reference_model.module.vae.cont_cov_type,
            n_hidden_cont_embed=reference_model.module.vae.n_hidden_cont_embed,
            n_layers_cont_embed=reference_model.module.vae.n_layers_cont_embed,
            ignore_covariates=ignore_covariates + reference_model.classification
            + reference_model.regression
            + reference_model.ordinal_regression,
        )

        vae.module.load_state_dict(reference_model.module.vae.state_dict())
        vae.to_device(device)

        new_vae = MultiVAE.load_query_data(
            adata,
            reference_model=vae,
            use_gpu=use_gpu,
            freeze=freeze,
            ignore_covariates=reference_model.classification
            + reference_model.regression
            + reference_model.ordinal_regression,
        )

        # model.module = reference_model.module
        model.module.vae = new_vae.module

        model.to_device(device)

        # if there are no prediction labels in the query
        if freeze is True:
            for name, p in model.module.named_parameters():
                if "vae" not in name:
                    p.requires_grad = False
        # if there are prediction labels in the query
        # if use_prediction_labels is True:
        #     new_state_dict = model.module.state_dict()
        #     for key, load_ten in load_state_dict.items():  # load_state_dict = old
        #         if 'vae' in key: # already updated
        #             load_state_dict[key] = new_state_dict[key]
        #         else: # MIL part
        #             new_ten = new_state_dict[key]
        #             if new_ten.size() == load_ten.size():
        #                 continue
        #             # new categoricals changed size
        #             else:
        #                 old_shape = new_ten.shape
        #                 new_shape = load_ten.shape
        #                 if old_shape[0] == new_shape[0]:
        #                     dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
        #                     fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
        #                 else:
        #                     dim_diff = new_ten.size()[0] - load_ten.size()[0]
        #                     fixed_ten = torch.cat([load_ten, new_ten[-dim_diff:, ...]], dim=0)
        #             load_state_dict[key] = fixed_ten

        #     model.module.load_state_dict(load_state_dict)

        #         # unfreeze last classifier layer
        #     model.module.classifiers[-1].weight.requires_grad = True
        #     model.module.classifiers[-1].bias.requires_grad = True

        model.module.eval()
        model.is_trained_ = False
       
        return model
