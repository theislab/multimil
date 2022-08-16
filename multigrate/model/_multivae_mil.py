import torch
import scvi
import scipy
import logging
import pandas as pd

from ..nn import *
from ._multivae import MultiVAE
from ..module import MultiVAETorch_MIL
from ..dataloaders import GroupDataSplitter, GroupAnnDataLoader
from ..utils import create_df
from typing import List, Optional, Union
from scvi.model.base import BaseModelClass
from scvi.module.base import auto_move_data
from scvi.train._callbacks import SaveBestState
from scvi.train import TrainRunner
from ..train import MILTrainingPlan
from scvi.data._anndata import _setup_anndata
from sklearn.metrics import classification_report
from scvi import _CONSTANTS
from anndata import AnnData
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base._utils import _initialize_model
from copy import deepcopy
from scvi.data import transfer_anndata_setup

logger = logging.getLogger(__name__)


class MultiVAE_MIL(BaseModelClass):
    def __init__(
        self,
        adata,
        modality_lengths,
        patient_label,
        classification=[],
        regression=[],
        ordinal_regression=[],
        patient_batch_size=128,
        integrate_on=None,
        condition_encoders=False,
        condition_decoders=True,
        normalization="layer",
        n_layers_encoders=[],
        n_layers_decoders=[],
        n_layers_shared_decoder: int = 1,
        n_hidden_encoders=[],
        n_hidden_decoders=[],
        n_hidden_shared_decoder: int = 32,
        add_patient_to_classifier=False,
        hierarchical_attn=True,
        add_shared_decoder=False,
        z_dim=16,
        h_dim=32,
        losses=[],
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
        n_layers_mlp_attn=None,
        n_layers_cont_embed: int = 1,
        n_hidden_cell_aggregator: int = 128,
        n_hidden_cov_aggregator: int = 128,
        n_hidden_classifier: int = 128,
        n_hidden_cont_embed: int = 128,
        n_hidden_mlp_attn=None,
        n_hidden_regressor: int = 128,
        attention_dropout=False,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        reg_coef=1.0,  # regularization
        regularize_cell_attn=False,
        regularize_cov_attn=False,
        regularize_vae=False,
        cont_cov_type="logsigm",
        drop_attn=False,
        mmd="latent",
        patient_in_vae=True,
        aggr="attn", # or 'both' = attn + average (two heads)
        cov_aggr=None, # one of ['attn', 'concat', 'both', 'mean']
    ):
        super().__init__(adata)

        self.patient_column = patient_label
        patient_idx = adata.uns["_scvi"]["extra_categoricals"]["keys"].index(
            patient_label
        )
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
                'When using aggr = "attn", cov_aggr has to be set to "concat", but cov_aggr={cov_aggr} was passed.'
            )

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
        if adata.uns["_scvi"].get("extra_continuous_keys") is not None:
            for i, key in enumerate(adata.uns["_scvi"]["extra_continuous_keys"]):
                if key != "size_factors":
                    if key in self.regression:
                        self.regression_idx.append(i)
                    else:
                        cont_covariate_dims.append(1)

        num_groups = 1
        integrate_on_idx = None
        if integrate_on:
            try:
                num_groups = len(
                    adata.uns["_scvi"]["extra_categoricals"]["mappings"][integrate_on]
                )
                integrate_on_idx = adata.uns["_scvi"]["extra_categoricals"][
                    "keys"
                ].index(integrate_on)
            except:
                raise ValueError(
                    f'Cannot integrate on {integrate_on}, has to be one of extra categoricals = {adata.uns["_scvi"]["extra_categoricals"]["keys"]}'
                )

        # classification and ordinal regression together here as ordinal regression values need to be registered as categorical covariates
        self.class_idx, self.ord_idx = [], []
        cat_covariate_dims = []
        num_classes = []
        if adata.uns["_scvi"].get("extra_categoricals") is not None:
            for i, key in enumerate(adata.uns["_scvi"]["extra_categoricals"]["keys"]):
                if key in self.classification:
                    num_classes.append(
                        adata.uns["_scvi"]["extra_categoricals"]["n_cats_per_key"][i]
                    )
                    self.class_idx.append(i)
                elif key in self.ordinal_regression:
                    self.ord_idx.append(i)
                else:  # the actual categorical covariate
                    if (key == self.patient_column and self.patient_in_vae) or (key != self.patient_column):
                        cat_covariate_dims.append(
                            adata.uns["_scvi"]["extra_categoricals"]["n_cats_per_key"][i]
                        )


        for label in ordinal_regression:
            print(
                f'The order for {label} ordinal classes is: {adata.obs[label].cat.categories}. If you need to change the order, please rerun setup_anndata and specify the correct order with "ordinal_regression_order" parameter.'
            )

        self.module = MultiVAETorch_MIL(
            modality_lengths=modality_lengths,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            z_dim=z_dim,
            h_dim=h_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            num_groups=num_groups,
            integrate_on_idx=integrate_on_idx,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_shared_decoder=n_layers_shared_decoder,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_shared_decoder=n_hidden_shared_decoder,
            n_hidden_cont_embed=n_hidden_cont_embed,
            add_shared_decoder=add_shared_decoder,
            # mil specific
            num_classes=num_classes,
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
            regularize_vae=regularize_vae,
            attention_dropout=attention_dropout,
            class_idx=self.class_idx,
            ord_idx=self.ord_idx,
            reg_idx=self.regression_idx,
            drop_attn=drop_attn,
            mmd=mmd,
            patient_in_vae=patient_in_vae,
            aggr=aggr,
            cov_aggr=cov_aggr,
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
        max_epochs: int = 500,
        lr: float = 1e-4,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        plan_kwargs: Optional[dict] = None,
        early_stopping_monitor: Optional[str] = "accuracy_validation",
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
        n_epochs_kl_warmup = max(max_epochs // 3, 1)
        update_dict = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            n_steps_kl_warmup=n_steps_kl_warmup,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping=early_stopping,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_patience=50,
            optimizer="AdamW",
            scale_adversarial_loss=1,
        )
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if save_best:
            if "callbacks" not in kwargs.keys():
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(
                SaveBestState(monitor="reconstruction_loss_validation")
            )
        data_splitter = GroupDataSplitter(
            self.adata,
            group_column=self.patient_column,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = MILTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            **kwargs,
        )
        return runner()

    def setup_anndata(
        adata,
        rna_indices_end=None,
        categorical_covariate_keys=None,
        continuous_covariate_keys=None,
        ordinal_regression_order=None,
    ):
        if rna_indices_end:
            if scipy.sparse.issparse(adata.X):
                adata.obs["size_factors"] = (
                    adata[:, :rna_indices_end].X.A.sum(1).T.tolist()
                )
            else:
                adata.obs["size_factors"] = (
                    adata[:, :rna_indices_end].X.sum(1).T.tolist()
                )

            if continuous_covariate_keys:
                continuous_covariate_keys.append("size_factors")
            else:
                continuous_covariate_keys = ["size_factors"]

        if ordinal_regression_order:
            if not set(ordinal_regression_order.keys()).issubset(
                categorical_covariate_keys
            ):
                raise ValueError(
                    f"All keys {ordinal_regression_order.keys()} has to be registered as categorical covariates too."
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

        return _setup_anndata(
            adata,
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
        )

    def setup_query(
        adata,
        query,
        rna_indices_end=None,
        categorical_covariate_keys=None,
        continuous_covariate_keys=None,
        ordinal_regression_order=None,
    ):
        MultiVAE_MIL.setup_anndata(
            query,
            rna_indices_end=rna_indices_end,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            ordinal_regression_order=ordinal_regression_order,
        )
        scvi.data.transfer_anndata_setup(
            adata,
            query,
            extend_categories=True,
        )

    @auto_move_data
    def get_model_output(
        self,
        adata=None,
        batch_size=256,
    ):
        with torch.no_grad():
            self.module.eval()
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

                cont_key = _CONSTANTS.CONT_COVS_KEY
                cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

                cat_key = _CONSTANTS.CAT_COVS_KEY
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

                if self.hierarchical_attn:
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
            if self.hierarchical_attn:
                adata.obsm["cov_attn"] = cov_level
            adata.obs["cell_attn"] = cell_level

            for i in range(len(self.class_idx)):
                name = self.classification[i]
                classes = adata.uns["_scvi"]["extra_categoricals"]["mappings"][name]
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

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
        freeze: bool = True,
    ):
        attr_dict = reference_model._get_user_attributes()
        attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}

        model = _initialize_model(cls, adata, attr_dict)

        scvi_setup_dict = attr_dict.pop("scvi_setup_dict_")
        transfer_anndata_setup(scvi_setup_dict, adata, extend_categories=True)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        ignore_categories = []
        if not reference_model.patient_in_vae:
            ignore_categories = [reference_model.patient_column]

        vae = MultiVAE(
            reference_model.adata,
            modality_lengths=reference_model.module.vae.input_dims,
            losses=reference_model.module.vae.losses,
            loss_coefs=reference_model.module.vae.loss_coefs,
            integrate_on=reference_model.module.vae.integrate_on_idx,
            condition_encoders=reference_model.module.vae.condition_encoders,
            condition_decoders=reference_model.module.vae.condition_decoders,
            normalization=reference_model.module.vae.normalization,
            z_dim=reference_model.module.vae.z_dim,
            h_dim=reference_model.module.vae.h_dim,
            dropout=reference_model.module.vae.dropout,
            cond_dim=reference_model.module.vae.cond_dim,
            kernel_type=reference_model.module.vae.kernel_type,
            n_layers_encoders=reference_model.module.vae.n_layers_encoders,
            n_layers_decoders=reference_model.module.vae.n_layers_decoders,
            n_layers_shared_decoder=reference_model.module.vae.n_layers_shared_decoder,
            n_hidden_encoders=reference_model.module.vae.n_hidden_encoders,
            n_hidden_decoders=reference_model.module.vae.n_hidden_decoders,
            n_hidden_shared_decoder=reference_model.module.vae.n_hidden_shared_decoder,
            add_shared_decoder=reference_model.module.vae.add_shared_decoder,
            cont_cov_type=reference_model.module.vae.cont_cov_type,
            n_hidden_cont_embed=reference_model.module.vae.n_hidden_cont_embed,
            n_layers_cont_embed=reference_model.module.vae.n_layers_cont_embed,
            ignore_categories=ignore_categories + reference_model.classification
            + reference_model.regression
            + reference_model.ordinal_regression,
        )

        vae.module.load_state_dict(reference_model.module.vae.state_dict())

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

        use_gpu, device = parse_use_gpu_arg(use_gpu)
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
        max_epochs: int = 500,
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
        n_steps_kl_warmup: Optional[int] = None,
        plan_kwargs: Optional[dict] = None,
        plot_losses=True,
        save_loss=None,
    ):
        ignore_categories = []
        if not self.patient_in_vae:
            ignore_categories = [self.patient_column]
        vae = MultiVAE(
            self.adata,
            modality_lengths=self.module.vae.input_dims,
            losses=self.module.vae.losses,
            loss_coefs=self.module.vae.loss_coefs,
            integrate_on=self.module.vae.integrate_on_idx,
            condition_encoders=self.module.vae.condition_encoders,
            condition_decoders=self.module.vae.condition_decoders,
            normalization=self.module.vae.normalization,
            z_dim=self.module.vae.z_dim,
            h_dim=self.module.vae.h_dim,
            dropout=self.module.vae.dropout,
            cond_dim=self.module.vae.cond_dim,
            kernel_type=self.module.vae.kernel_type,
            n_layers_encoders=self.module.vae.n_layers_encoders,
            n_layers_decoders=self.module.vae.n_layers_decoders,
            n_layers_shared_decoder=self.module.vae.n_layers_shared_decoder,
            n_hidden_encoders=self.module.vae.n_hidden_encoders,
            n_hidden_decoders=self.module.vae.n_hidden_decoders,
            n_hidden_shared_decoder=self.module.vae.n_hidden_shared_decoder,
            add_shared_decoder=self.module.vae.add_shared_decoder,
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
            n_steps_kl_warmup=n_steps_kl_warmup,
            plan_kwargs=plan_kwargs,
        )
        if plot_losses:
            vae.plot_losses(save=save_loss)

        self.module.vae = vae.module
        self.is_trained_ = True
