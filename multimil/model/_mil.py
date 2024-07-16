import torch
import logging
import pandas as pd
import anndata as ad
import numpy as np
import warnings
from matplotlib import pyplot as plt

from ..dataloaders import GroupDataSplitter, GroupAnnDataLoader
from ..module import MILClassifierTorch
from ..utils import create_df, setup_ordinal_regression
from typing import List, Optional, Union, Dict
from math import ceil
from scvi.model.base import BaseModelClass, ArchesMixin
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.train._callbacks import SaveBestState
from scvi.train import TrainRunner, AdversarialTrainingPlan
from scvi import REGISTRY_KEYS
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.data import AnnDataManager, fields
from anndata import AnnData
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base._utils import _initialize_model
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

class MILClassifier(BaseModelClass, ArchesMixin):
    def __init__(
        self,
        adata,
        sample_key,
        classification=[],
        regression=[],
        ordinal_regression=[],
        sample_batch_size=128,
        normalization="layer",
        z_dim=16,
        dropout=0.2,
        scoring="gated_attn",
        attn_dim=16,
        n_layers_cell_aggregator: int = 1,
        n_layers_classifier: int = 2,
        n_layers_regressor: int = 2,
        n_layers_mlp_attn: int = 1,
        n_hidden_cell_aggregator: int = 128,
        n_hidden_classifier: int = 128,
        n_hidden_mlp_attn: int = 32,
        n_hidden_regressor: int = 128,
        attention_dropout=False,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        drop_attn=False,
        aggr="attn", # or 'both' = attn + average (two heads)
        activation='leaky_relu', # or tanh
        initialization=None, # xavier (tanh) or kaiming (leaky_relu)
        weighted_class_loss=False, 
        anneal_class_loss=False,
        ignore_covariates=None,
    ):
        super().__init__(adata)

        self.sample_key = sample_key
        self.scoring = scoring

        self.sample_idx = None
        if self.sample_key not in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
                raise ValueError(
                    f"Sample key = '{self.sample_key}' has to be one of the registered categorical covariates = {self.adata_manager.registry['setup_args']['categorical_covariate_keys']}"
                )
        self.sample_idx = self.adata_manager.registry["setup_args"]["categorical_covariate_keys"].index(self.sample_key)
       
        if len(classification) + len(regression) + len(ordinal_regression) == 0:
            raise ValueError(
                'At least one of "classification", "regression", "ordinal_regression" has to be specified.'
            )

        self.classification = classification
        self.regression = regression
        self.ordinal_regression = ordinal_regression

        # TODO check if all of the three above were registered with setup anndata
        # TODO add check that class is the same within a patient
        # TODO assert length of things is the same as number of modalities
        # TODO add that n_layers has to be > 0 for all
        # TODO warning if n_layers == 1 then n_hidden is not used for classifier and MLP attention
        # TODO warning if MLP attention is used but n layers and n hidden not given that using default values
        if scoring == "MLP":
            if not n_layers_mlp_attn:
                n_layers_mlp_attn = 1
            if not n_hidden_mlp_attn:
                n_hidden_mlp_attn = 16
        
        self.regression_idx = []
        if len(cont_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)) > 0:
            for key in cont_covs['columns']:
                if key in self.regression:
                    self.regression_idx.append(
                        list(cont_covs['columns']).index(key)
                    )
                else: # only can happen when using multivae_mil
                    if ignore_covariates is not None and key not in ignore_covariates:
                        warnings.warn(
                            f"Registered continuous covariate '{key}' is not in regression covariates so will be ignored."
                        )
       
        # classification and ordinal regression together here as ordinal regression values need to be registered as categorical covariates
        self.class_idx, self.ord_idx = [], []
        self.num_classification_classes = []
        if len(cat_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)) > 0:
            for i, num_cat in enumerate(cat_covs.n_cats_per_key):
                cat_cov_name = cat_covs['field_keys'][i]
                if cat_cov_name in self.classification:
                    self.num_classification_classes.append(
                        num_cat
                    )
                    self.class_idx.append(i)
                elif cat_cov_name in self.ordinal_regression:
                    self.num_classification_classes.append(
                        num_cat
                    )
                    self.ord_idx.append(i)
                else:  # the actual categorical covariate, only can happen when using multivae_mil
                    if ignore_covariates is not None and cat_cov_name not in ignore_covariates and cat_cov_name != self.sample_key:
                        warnings.warn(
                            f"Registered categorical covariate '{cat_cov_name}' is not in classification or ordinal regression covariates and is not the sample covariate so will be ignored."
                        )
                        
        for label in ordinal_regression:
            print(
                f'The order for {label} ordinal classes is: {adata.obs[label].cat.categories}. If you need to change the order, please rerun setup_anndata and specify the correct order with the `ordinal_regression_order` parameter.'
            )

        # TODO probably remove
        # create a list with a dict per classification label with weights per class for that label
        self.class_weights = None
        if weighted_class_loss is True:
            for label in self.classification:
                self.class_weights_dict = dict(adata.obsm['_scvi_extra_categorical_covs'][label].value_counts())
                denominator = 0.0
                for _, n_obs_in_class in self.class_weights_dict.items():
                    denominator += (1.0 / n_obs_in_class)
                self.class_weights.append({name: (1 / value ) / denominator for name, value in self.class_weights_dict.items()})

        self.class_idx = torch.tensor(self.class_idx)
        self.ord_idx = torch.tensor(self.ord_idx)
        self.regression_idx = torch.tensor(self.regression_idx)

        self.module = MILClassifierTorch(
            z_dim=z_dim,
            dropout=dropout,
            activation=activation,
            initialization=initialization,
            normalization=normalization,
            num_classification_classes=self.num_classification_classes,
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
            sample_idx=self.sample_idx,
            sample_batch_size=sample_batch_size,
            attention_dropout=attention_dropout,
            class_idx=self.class_idx,
            ord_idx=self.ord_idx,
            reg_idx=self.regression_idx,
            drop_attn=drop_attn,
            aggr=aggr,
            class_weights=self.class_weights,
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
        early_stopping_monitor: Optional[str] = "accuracy_validation",
        early_stopping_mode: Optional[str] = "max",
        save_checkpoint_every_n_epochs: Optional[int] = None,
        path_to_checkpoints: Optional[str] = None,
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
        if len(self.regression) > 0:
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
            group_column=self.sample_key,
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

        setup_ordinal_regression(adata, ordinal_regression_order, categorical_covariate_keys)

        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer=None,),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, None),
            fields.CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            fields.NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]

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
            group_column=self.sample_key,
            drop_last=False,
        )

        cell_level_attn, bags = [], []
        class_pred, ord_pred, reg_pred = {}, {}, {}
        (
            bag_class_true,
            bag_class_pred,
            bag_reg_true,
            bag_reg_pred,
            bag_ord_true,
            bag_ord_pred,
        ) = ({}, {}, {}, {}, {}, {})

        batch_start_idx = 0
        batch_end_idx = 0

        bag_counter = 0
        s = 0 # TODO rename
        for tensors in scdl:

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            actual_batch_size = cat_covs.shape[0]
            batch_end_idx += actual_batch_size

            idx = list(
                range(
                    self.module.sample_batch_size,
                    actual_batch_size,
                    self.module.sample_batch_size,
                )
            ) 
            # these batches can be any size between 1 and batch_size, this is intended
            # as if we split them into 128, 128, 2 e.g., then stacking and forward pass would not work
            if (
                actual_batch_size % self.module.sample_batch_size != 0
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
            pred = outputs["predictions"]

            num_of_bags_in_batch = pred[0].shape[0]
            
            if len(idx) == 0:
                bags += [[bag_counter] * actual_batch_size]
                s += actual_batch_size
                bag_counter += 1
            else:
                bags += [[bag_counter + i]*self.module.sample_batch_size for i in range(num_of_bags_in_batch)]
                bag_counter += num_of_bags_in_batch
                s += self.module.sample_batch_size * num_of_bags_in_batch
            cell_attn = self.module.cell_level_aggregator[-1].A.squeeze(dim=1)
            size = cell_attn.shape[-1]
            cell_attn = (
                cell_attn.flatten()
            )  # in inference always one patient per batch

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

            cell_level_attn += [cell_attn.cpu()]
            batch_start_idx += actual_batch_size
        
        cell_level = torch.cat(cell_level_attn).numpy()
        
        adata.obs["cell_attn"] = cell_level
        flat_bags = [value for sublist in bags for value in sublist]
        adata.obs["bags"] = flat_bags

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
            adata.uns[f"bag_classification_true_{name}"] = create_df(bag_class_true, self.classification)
            df_bag = create_df(bag_class_pred[i], classes)
            adata.uns[f"bag_classification_predictions_{name}"] = df_bag
            adata.uns[f"bag_predicted_{name}"] = df_bag.to_numpy().argmax(axis=1)
        # TODO ord regression only tested with one label
        for i in range(len(self.ord_idx)):
            name = self.ordinal_regression[i]
            classes = self.adata_manager.get_state_registry('extra_categorical_covs')['mappings'][name]
            df = create_df(ord_pred[i], columns=['pred_regression_value'], index=adata.obs_names)
            adata.obsm[f"ord_regression_predictions_{name}"] = df
            adata.obs[f"predicted_{name}"] = np.clip(np.round(df.to_numpy()), a_min=0.0, a_max=len(classes) - 1.0)
            adata.obs[f"predicted_{name}"] = adata.obs[f"predicted_{name}"].astype(int).astype("category")
            adata.obs[f"predicted_{name}"] = adata.obs[
                f"predicted_{name}"
            ].cat.rename_categories({i: cl for i, cl in enumerate(classes)})
            adata.uns[f"bag_ord_regression_true_{name}"] = create_df(bag_ord_true, self.ordinal_regression)
            df_bag = create_df(bag_ord_pred[i], ['predicted_regression_value'])
            adata.uns[f"bag_ord_regression_predictions_{name}"] = df_bag
            adata.uns[f"bag_predicted_{name}"] = np.clip(np.round(df_bag.to_numpy()), a_min=0.0, a_max=len(classes) - 1.0)

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

    def plot_losses(self, save=None):
        """Plot losses."""
        df = pd.DataFrame(self.history["train_loss_epoch"])
        for key in self.history.keys():
            if key != "train_loss_epoch":
                df = df.join(self.history[key])

        df["epoch"] = df.index

        loss_names = []

        if self.module.class_loss_coef != 0 and len(self.module.class_idx) > 0:
            loss_names.extend(["class_loss", "accuracy"])
        
        if self.module.regression_loss_coef != 0 and len(self.module.reg_idx) > 0:
            loss_names.append("regression_loss")

        if self.module.regression_loss_coef != 0 and len(self.module.ord_idx) > 0:
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

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/model/base/_archesmixin.py#L30
    # accessed on 7 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
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

        # TODO I think currently this function works only if the prediction cov is present in the .obs of the query
        # need to allow it to be missing

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
        model.to_device(device)

        model.module.eval()
        model.is_trained_ = True

        return model
