import torch
import scvi
import scipy
import logging
import pandas as pd
import anndata as ad
import multigrate as mtg
from matplotlib import pyplot as plt

from multigrate.model import MultiVAE
from multigrate.dataloaders import GroupDataSplitter, GroupAnnDataLoader
from ..module import MILClassifierTorch
from ..utils import create_df
from typing import List, Optional, Union, Dict
from math import ceil
from scvi.model.base import BaseModelClass
from scvi.module.base import auto_move_data
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.train._callbacks import SaveBestState
from scvi.train import TrainRunner, AdversarialTrainingPlan
from sklearn.metrics import classification_report
from scvi import REGISTRY_KEYS
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.data import AnnDataManager, fields
from anndata import AnnData
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base._utils import _initialize_model
from copy import deepcopy

logger = logging.getLogger(__name__)


class MILClassifier(BaseModelClass):
    def __init__(
        self,
        adata,
        patient_label,
        classification=[],
        regression=[],
        ordinal_regression=[],
        patient_batch_size=128,
        integrate_on=None,
        condition_encoders=False,
        condition_decoders=True,
        normalization="layer",
        n_layers_encoders=None,
        n_layers_decoders=None,
        n_hidden_encoders=None,
        n_hidden_decoders=None,
        add_patient_to_classifier=False,
        hierarchical_attn=True,
        z_dim=16,
        losses=None,
        dropout=0.2,
        cond_dim=15,
        kernel_type="gaussian",
        loss_coefs=[],
        scoring="gated_attn",
        attn_dim=16,
        n_layers_cell_aggregator: int = 1,
        n_layers_cov_aggregator: int = 1,
        n_layers_classifier: int = 1,
        n_layers_regressor: int = 1,
        n_layers_mlp_attn: int = 1,
        n_layers_cont_embed: int = 1,
        n_hidden_cell_aggregator: int = 128,
        n_hidden_cov_aggregator: int = 128,
        n_hidden_classifier: int = 128,
        n_hidden_cont_embed: int = 128,
        n_hidden_mlp_attn: int = 32,
        n_hidden_regressor: int = 128,
        attention_dropout=False,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        reg_coef=1.0,  # regularization
        regularize_cell_attn=False,
        regularize_cov_attn=False,
        cont_cov_type="logsigm",
        drop_attn=False,
        mmd="latent",
        patient_in_vae=True,
        aggr="attn", # or 'both' = attn + average (two heads)
        cov_aggr=None, # one of ['attn', 'concat', 'both', 'mean']
        activation='leaky_relu', # or tanh
        initialization=None, # xavier (tanh) or kaiming (leaky_relu)
        weighted_class_loss=False, 
        anneal_class_loss=False,
    ):
        super().__init__(adata)

        modality_lengths = [adata.uns["modality_lengths"][key] for key in sorted(adata.uns["modality_lengths"].keys())]
        if losses is None:
            losses = ["mse"]*len(modality_lengths)
        if ("nb" in losses or "zinb" in losses) and REGISTRY_KEYS.SIZE_FACTOR_KEY not in self.adata_manager.data_registry:
            raise ValueError(f"Have to register {REGISTRY_KEYS.SIZE_FACTOR_KEY} when using 'nb' or 'zinb' loss.")

        self.patient_column = patient_label

        patient_idx = None
        if self.patient_column not in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
                raise ValueError(
                    f"Patient label = '{self.patient_column}' has to be one of the registered categorical covariates = {self.adata_manager.registry['setup_args']['categorical_covariate_keys']}"
                )
        patient_idx = self.adata_manager.registry["setup_args"]["categorical_covariate_keys"].index(self.patient_column)

        self.patient_in_vae = patient_in_vae
        self.scoring = scoring
        self.adata = adata
        self.hierarchical_attn = hierarchical_attn
       
       

        if len(classification) + len(regression) + len(ordinal_regression) == 0:
            raise ValueError(
                'At least one of "classification", "regression", "ordinal_regression" has to be specified.'
            )

        self.classification = classification
        self.regression = regression
        self.ordinal_regression = ordinal_regression

        if cov_aggr is None:
            if aggr == 'attn':
                cov_aggr = 'attn'
            else: # aggr = 'both'
                cov_aggr = 'concat'
        else:
            if aggr == 'both' and cov_aggr != 'concat':
                raise ValueError(
                'When using aggr = "both", cov_aggr has to be set to "concat", but cov_aggr={cov_aggr} was passed.'
            )
        self.cov_aggr = cov_aggr

        # TODO check if all of the three above were registered with setup anndata
        # TODO add check that class is the same within a patient
        # TODO assert length of things is the same as number of modalities
        # TODO add that n_layers has to be > 0 for all
        # TODO warning if n_layers == 1 then n_hidden is not used for classifier and MLP attention
        # TODO warning if MLP attention is used but n layers and n hidden not given that using default values
        # TODO if aggr='both' and hierarchical_attn=True then cov_aggr has to be 'concat'
        if scoring == "MLP":
            if not n_layers_mlp_attn:
                n_layers_mlp_attn = 1
            if not n_hidden_mlp_attn:
                n_hidden_mlp_attn = 16

        self.regression_idx = []

        cont_covariate_dims = []
        if len(cont_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)) > 0:
            cont_covariate_dims = [
                1
                for key in cont_covs['columns']
                if key not in self.regression
            ]

        num_groups = 1
        integrate_on_idx = None
        if integrate_on is not None:
            if integrate_on not in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
                raise ValueError(
                    f"Cannot integrate on '{integrate_on}', has to be one of the registered categorical covariates = {self.adata_manager.registry['setup_args']['categorical_covariate_keys']}"
                )
            elif integrate_on in self.classification:
                raise ValueError(
                    f"Specified integrate_on = '{integrate_on}' is in classification covariates = {self.classification}."
                )
            elif integrate_on in self.ordinal_regression:
                raise ValueError(
                    f"Specified integrate_on = '{integrate_on}' is in ordinal regression covariates = {self.ordinal_regression}."
                )
            else:
                num_groups = len(self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)['mappings'][integrate_on])
                integrate_on_idx = self.adata_manager.registry["setup_args"]["categorical_covariate_keys"].index(integrate_on)

        # classification and ordinal regression together here as ordinal regression values need to be registered as categorical covariates
        self.class_idx, self.ord_idx = [], []
        cat_covariate_dims = []
        num_classification_classes = []
        if len(cat_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)) > 0:
            for i, num_cat in enumerate(cat_covs.n_cats_per_key):
                cat_cov_name = cat_covs['field_keys'][i]
                if cat_cov_name in self.classification:
                    num_classification_classes.append(
                        num_cat
                    )
                    self.class_idx.append(i)
                elif cat_cov_name in self.ordinal_regression:
                    self.ord_idx.append(i)
                else:  # the actual categorical covariate
                    if (cat_cov_name == self.patient_column and self.patient_in_vae) or (cat_cov_name != self.patient_column):
                        cat_covariate_dims.append(
                            num_cat
                        )

        for label in ordinal_regression:
            print(
                f'The order for {label} ordinal classes is: {adata.obs[label].cat.categories}. If you need to change the order, please rerun setup_anndata and specify the correct order with the `ordinal_regression_order` parameter.'
            )

        # create a list with a dict per classification label with weights per class for that label
        class_weights = None
        if weighted_class_loss is True:
            for label in self.classification:
                class_weights_dict = dict(adata.obsm['_scvi_extra_categorical_covs'][label].value_counts())
                denominator = 0.0
                for _, n_obs_in_class in class_weights_dict.items():
                    denominator += (1.0 / n_obs_in_class)
                class_weights.append({name: (1 / value ) / denominator for name, value in class_weights_dict.items()})

        self.module = MILClassifierTorch(
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
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_cont_embed=n_hidden_cont_embed,
            activation=activation,
            initialization=initialization,
            # mil specific
            num_classification_classes=num_classification_classes,
            scoring=scoring,
            attn_dim=attn_dim,
            cat_covariate_dims=cat_covariate_dims,
            cont_covariate_dims=cont_covariate_dims,
            cont_cov_type=cont_cov_type,
            n_layers_cell_aggregator=n_layers_cell_aggregator,
            n_layers_cov_aggregator=n_layers_cov_aggregator,
            n_layers_classifier=n_layers_classifier,
            n_layers_mlp_attn=n_layers_mlp_attn,
            n_layers_regressor=n_layers_regressor,
            n_hidden_regressor=n_hidden_regressor,
            n_hidden_cell_aggregator=n_hidden_cell_aggregator,
            n_hidden_cov_aggregator=n_hidden_cov_aggregator,
            n_hidden_classifier=n_hidden_classifier,
            n_hidden_mlp_attn=n_hidden_mlp_attn,
            class_loss_coef=class_loss_coef,
            regression_loss_coef=regression_loss_coef,
            reg_coef=reg_coef,
            add_patient_to_classifier=add_patient_to_classifier,
            patient_idx=patient_idx,
            hierarchical_attn=hierarchical_attn,
            patient_batch_size=patient_batch_size,
            regularize_cell_attn=regularize_cell_attn,
            regularize_cov_attn=regularize_cov_attn,
            attention_dropout=attention_dropout,
            class_idx=self.class_idx,
            ord_idx=self.ord_idx,
            reg_idx=self.regression_idx,
            drop_attn=drop_attn,
            mmd=mmd,
            patient_in_vae=patient_in_vae,
            aggr=aggr,
            cov_aggr=cov_aggr,
            class_weights=class_weights,
            anneal_class_loss=anneal_class_loss,
        )

        self.class_idx = torch.tensor(self.class_idx)
        self.ord_idx = torch.tensor(self.ord_idx)
        self.regression_idx = torch.tensor(self.regression_idx)

        self.init_params_ = self._get_init_params(locals())

    # TODO discuss if we still need it
    def use_model(self, model, freeze_vae=True, freeze_cov_embeddings=True):
        state_dict = model.module.state_dict()
        self.module.vae.load_state_dict(state_dict)
        if freeze_vae:
            for key, p in self.module.vae.named_parameters():
                p.requires_grad = False
        if not freeze_cov_embeddings:
            for embed in self.module.vae.cat_covariate_embeddings:
                for _, p in embed.named_parameters():
                    p.requires_grad = True
            for embed in self.module.vae.cont_covariate_embeddings:
                for _, p in embed.named_parameters():
                    p.requires_grad = True

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

        if self.patient_column is not None:
            data_splitter = GroupDataSplitter(
                self.adata_manager,
                group_column=self.patient_column,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )

        if self.patient_column is not None:
            data_splitter = GroupDataSplitter(
                self.adata_manager,
                group_column=self.patient_column,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
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

        setup_method_args = cls._get_setup_method_args(**locals())

        batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer=None,),
            batch_field,
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
            group_column=self.patient_column,
            drop_last=False,
        )

        latent, cell_level_attn, cov_level_attn = [], [], []
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
                    self.module.patient_batch_size,
                    batch_size,
                    self.module.patient_batch_size,
                )
            )  # or depending on model.train() and model.eval() ???
            if (
                batch_size % self.module.patient_batch_size != 0
            ):  # can only happen during inference for last batches for each patient
                idx = []

            if len(self.regression_idx) > 0:
                regression = torch.index_select(cont_covs, 1, self.regression_idx)
                regression = regression.view(
                    len(idx) + 1, -1, len(self.regression_idx)
                )[:, 0, :]

            if len(self.ord_idx) > 0:
                ordinal_regression = torch.index_select(cat_covs, 1, self.ord_idx)
                ordinal_regression = ordinal_regression.view(
                    len(idx) + 1, -1, len(self.ord_idx)
                )[:, 0, :]
            if len(self.class_idx) > 0:
                classification = torch.index_select(cat_covs, 1, self.class_idx)
                classification = classification.view(
                    len(idx) + 1, -1, len(self.class_idx)
                )[:, 0, :]

            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z_joint"]
            pred = outputs["predictions"]
            cell_attn = self.module.cell_level_aggregator[-1].A.squeeze(dim=1)
            size = cell_attn.shape[-1]
            cell_attn = (
                cell_attn.flatten()
            )  # in inference always one patient per batch

            if self.hierarchical_attn and self.cov_aggr in ["attn", "both"]:
                cov_attn = self.module.cov_level_aggregator[-1].A.squeeze(
                    dim=1
                )  # aggregator is always last after hidden MLP layers
                cov_attn = cov_attn.unsqueeze(1).repeat(1, size, 1)
                cov_attn = cov_attn.flatten(start_dim=0, end_dim=1)
                cov_level_attn += [cov_attn.cpu()]

            for i in range(len(self.class_idx)):
                bag_class_pred[i] = bag_class_pred.get(i, []) + [pred[i].cpu()]
                bag_class_true[i] = bag_class_true.get(i, []) + [
                    classification[:, i].cpu()
                ]
                class_pred[i] = class_pred.get(i, []) + [
                    pred[i].unsqueeze(1).repeat(1, size, 1).flatten(0, 1)
                ]
            for i in range(len(self.ord_idx)):
                bag_ord_pred[i] = bag_ord_pred.get(i, []) + [
                    pred[len(self.class_idx) + i]
                ]
                bag_ord_true[i] = bag_ord_true.get(i, []) + [
                    ordinal_regression[:, i].cpu()
                ]
                ord_pred[i] = ord_pred.get(i, []) + [
                    pred[len(self.class_idx) + i].repeat(1, size).flatten()
                ]
            for i in range(len(self.regression_idx)):
                bag_reg_pred[i] = bag_reg_pred.get(i, []) + [
                    pred[len(self.class_idx) + len(self.ord_idx) + i].cpu()
                ]
                bag_reg_true[i] = bag_reg_true.get(i, []) + [regression[:, i].cpu()]
                reg_pred[i] = reg_pred.get(i, []) + [
                    pred[len(self.class_idx) + len(self.ord_idx) + i]
                    .repeat(1, size)
                    .flatten()
                ]

            latent += [z.cpu()]
            cell_level_attn += [cell_attn.cpu()]

        if len(cov_level_attn) == 0:
            cov_level_attn = [torch.Tensor()]

        latent = torch.cat(latent).numpy()
        cell_level = torch.cat(cell_level_attn).numpy()
        cov_level = torch.cat(cov_level_attn).numpy()

        adata.obsm["latent"] = latent
        if self.hierarchical_attn and self.cov_aggr in ["attn", "both"]:
            adata.obsm["cov_attn"] = cov_level
        adata.obs["cell_attn"] = cell_level

        for i in range(len(self.class_idx)):
            name = self.classification[i]
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
                bag_class_true, self.classification
            )
            df_bag = create_df(bag_class_pred[i], classes)
            adata.uns[f"bag_classification_predictions_{name}"] = df_bag
            adata.uns[f"bag_predicted_{name}"] = df_bag.to_numpy().argmax(axis=1)
        # TODO fix ordinal regression and regression for multiple labels
        if len(self.ord_idx) > 0:
            adata.obsm["ordinal_predictions"] = create_df(
                ord_pred, self.ordinal_regression, index=adata.obs_names
            )
            adata.uns["bag_ordinal_true"] = create_df(
                bag_ord_true, self.ordinal_regression
            )
            adata.uns["bag_ordinal_predictions"] = create_df(
                bag_ord_pred, self.ordinal_regression
            )
        if len(self.regression_idx) > 0:
            adata.obsm["regression_predictions"] = create_df(
                reg_pred, self.regression, index=adata.obs_names
            )
            adata.uns["bag_regression_true"] = create_df(
                bag_reg_true, self.regression
            )
            adata.uns["bag_regression_predictions"] = create_df(
                bag_reg_pred, self.regression
            )

    # TODO fix with multiple classification labels
    def classification_report(self, label, adata=None, level="patient"):
        # TODO this works if classification now, do a custom one for ordinal and check what to report for regression
        adata = self._validate_anndata(adata)
        # TODO check if label is in cat covariates and was predicted
        if "predicted_class" not in adata.obs.keys():
            raise RuntimeError(
                f'"predicted_class" not in adata.obs.keys(), please run model.get_model_output(adata) first.'
            )

        target_names = adata.uns["_scvi"]["extra_categoricals"]["mappings"][label]

        if level == "cell":
            y_true = adata.obsm["_scvi_extra_categoricals"][label].values
            y_pred = adata.obs["predicted_class"].values  # TODO can be more now
            print(classification_report(y_true, y_pred, target_names=target_names))
        elif level == "bag":
            y_true = adata.uns["bag_true"]
            y_pred = adata.uns["bag_pred"]
            print(classification_report(y_true, y_pred, target_names=target_names))
        elif level == "patient":
            y_true = (
                pd.DataFrame(adata.obs[self.patient_column])
                .join(pd.DataFrame(adata.obsm["_scvi_extra_categoricals"][label]))
                .groupby(self.patient_column)
                .agg("first")
            )
            y_true = y_true[label].values
            y_pred = (
                adata.obs[[self.patient_column, "predicted_class"]]
                .groupby(self.patient_column)
                .agg(lambda x: x.value_counts().index[0])
            )
            y_pred = y_pred["predicted_class"].values
            print(classification_report(y_true, y_pred, target_names=target_names))
        else:
            raise RuntimeError(f"level={level} not in ['patient', 'bag', 'cell'].")

    def plot_losses(self, save=None):
        """Plot losses."""
        df = pd.DataFrame(self.history["train_loss_epoch"])
        for key in self.history.keys():
            if key != "train_loss_epoch":
                df = df.join(self.history[key])

        df["epoch"] = df.index

        loss_names = ["kl_local", "elbo", "reconstruction_loss"]

        if self.module.vae.loss_coefs["integ"] != 0:
            loss_names.append("integ_loss")

        if self.module.class_loss_coef != 0 and len(self.module.class_idx) > 0:
            loss_names.extend(["class_loss", "accuracy"])
        
        if self.module.regression_loss_coef != 0 and len(self.module.reg_idx) > 0:
            loss_names.append("regression_loss")

        if self.module.reg_coef != 0 and (self.module.regularize_cov_attn or self.module.regularize_cell_attn):
            loss_names.append("reg_loss")

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

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/model/base/_archesmixin.py#L30
    # accessed on 7 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
        freeze: bool = True,
    ):
        """Online update of a reference model with scArches algorithm # TODO cite.

        :param adata:
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
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

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        adata_manager = model.get_anndata_manager(adata, required=True)

        ignore_categories = []
        if not reference_model.patient_in_vae:
            ignore_categories = [reference_model.patient_column]

        ref_adata = reference_model.adata.copy()
        setup_args = reference_model.adata_manager.registry["setup_args"].copy()
        setup_args.pop('ordinal_regression_order')

        mtg.model.MultiVAE.setup_anndata(ref_adata, **setup_args)

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
            ignore_categories=ignore_categories + reference_model.classification
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
            ignore_categories=reference_model.classification
            + reference_model.regression
            + reference_model.ordinal_regression,
        )

        model.module = reference_model.module
        model.module.vae = new_vae.module

        model.to_device(device)

        if freeze:
            for name, p in model.module.named_parameters():
                if "vae" not in name:
                    p.requires_grad = False

        model.module.eval()
        model.is_trained_ = False

        return model

    def finetune_query(
        self,
        max_epochs: int = 200,
        lr: float = 1e-4,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        weight_decay: float = 0,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        adversarial_mixing: bool = False,
        plan_kwargs: Optional[dict] = None,
        plot_losses=True,
        save_loss=None,
        **kwargs,
    ):
        ignore_categories = []
        if not self.patient_in_vae:
            ignore_categories = [self.patient_column]
        vae = MultiVAE(
            self.adata,
            losses=self.module.vae.losses,
            loss_coefs=self.module.vae.loss_coefs,
            integrate_on=self.module.vae.integrate_on_idx,
            condition_encoders=self.module.vae.condition_encoders,
            condition_decoders=self.module.vae.condition_decoders,
            normalization=self.module.vae.normalization,
            z_dim=self.module.vae.z_dim,
            dropout=self.module.vae.dropout,
            cond_dim=self.module.vae.cond_dim,
            kernel_type=self.module.vae.kernel_type,
            n_layers_encoders=self.module.vae.n_layers_encoders,
            n_layers_decoders=self.module.vae.n_layers_decoders,
            n_hidden_encoders=self.module.vae.n_hidden_encoders,
            n_hidden_decoders=self.module.vae.n_hidden_decoders,
            cont_cov_type=self.module.vae.cont_cov_type,
            n_hidden_cont_embed=self.module.vae.n_hidden_cont_embed,
            n_layers_cont_embed=self.module.vae.n_layers_cont_embed,
            ignore_categories=ignore_categories + self.classification
            + self.regression
            + self.ordinal_regression,
        )

        vae.module.load_state_dict(self.module.vae.state_dict())

        for (frozen_name, frozen_p), (name, p) in zip(
            self.module.vae.named_parameters(), vae.module.named_parameters()
        ):
            assert frozen_name == name
            p.requires_grad = frozen_p.requires_grad

        vae.train(
            max_epochs=max_epochs,
            lr=lr,
            use_gpu=use_gpu,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            weight_decay=weight_decay,
            eps=eps,
            early_stopping=early_stopping,
            save_best=save_best,
            check_val_every_n_epoch=check_val_every_n_epoch,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            n_steps_kl_warmup=n_steps_kl_warmup,
            adversarial_mixing=adversarial_mixing,
            plan_kwargs=plan_kwargs,
            **kwargs,
        )
        if plot_losses:
            vae.plot_losses(save=save_loss)

        self.module.vae = vae.module
        self.is_trained_ = True
