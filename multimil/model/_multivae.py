import logging
from math import ceil
from typing import Dict, List, Literal, Optional, Union

import anndata as ad
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base import ArchesMixin, BaseModelClass
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.model.base._utils import _initialize_model
from scvi.train import AdversarialTrainingPlan, TrainRunner
from scvi.train._callbacks import SaveBestState

from ..dataloaders import GroupDataSplitter
from ..module import MultiVAETorch

logger = logging.getLogger(__name__)


class MultiVAE(BaseModelClass, ArchesMixin):
    """Multigrate model.

    :param adata:
        AnnData object that has been registered via :meth:`~multigrate.model.MultiVAE.setup_anndata`.
    :param integrate_on:
        One of the categorical covariates refistered with :math:`~multigrate.model.MultiVAE.setup_anndata` to integrate on. The latent space then will be disentangled from this covariate. If `None`, no integration is performed.
    :param condition_encoders:
        Whether to concatentate covariate embeddings to the first layer of the encoders. Default is `False`.
    :param condition_decoders:
        Whether to concatentate covariate embeddings to the first layer of the decoders. Default is `True`.
    :param normalization:
        What normalization to use; has to be one of `batch` or `layer`. Default is `layer`.
    :param z_dim:
        Dimensionality of the latent space. Default is 15.
    :param losses:
        Which losses to use for each modality. Has to be the same length as the number of modalities. Default is `MSE` for all modalities.
    :param dropout:
        Dropout rate. Default is 0.2.
    :param cond_dim:
        Dimensionality of the covariate embeddings. Default is 10.
    :param loss_coefs:
        Loss coeficients for the different losses in the model. Default is 1 for all.
    :param n_layers_encoders:
        Number of layers for each encoder. Default is 2 for all modalities. Has to be the same length as the number of modalities.
    :param n_layers_decoders:
        Number of layers for each decoder. Default is 2 for all modalities. Has to be the same length as the number of modalities.
    :param n_hidden_encoders:
        Number of nodes for each hidden layer in the encoders. Default is 32.
    :param n_hidden_decoders:
        Number of nodes for each hidden layer in the decoders. Default is 32.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        integrate_on: Optional[str] = None,
        condition_encoders: bool = False,
        condition_decoders: bool = True,
        normalization: Literal["layer", "batch", None] = "layer",
        z_dim: int = 15,
        losses: Optional[List[str]] = None,
        dropout: float = 0.2,
        cond_dim: int = 10,
        kernel_type: Literal["gaussian", None] = "gaussian",
        loss_coefs: Dict[int, float] = None,
        cont_cov_type: Literal["logsim", "sigm", None] = "logsigm",
        n_layers_cont_embed: int = 1,
        n_layers_encoders: Optional[List[int]] = None,
        n_layers_decoders: Optional[List[int]] = None,
        n_hidden_cont_embed: int = 32,
        n_hidden_encoders: Optional[List[int]] = None,
        n_hidden_decoders: Optional[List[int]] = None,
        ignore_categories: Optional[List[str]] = None,
        mmd: Literal["latent", "marginal", "both"] = "latent",
        activation: Optional[str] = "leaky_relu",
        initialization: Optional[str] = None,
    ):
        super().__init__(adata)

        self.adata = adata
        self.group_column = integrate_on

        # TODO: add options for number of hidden layers, hidden layers dim and output activation functions
        if normalization not in ["layer", "batch", None]:
            raise ValueError('Normalization has to be one of ["layer", "batch", None]')
        # TODO: do some assertions for other parameters

        if ignore_categories is None:
            ignore_categories = []

        if (
            "nb" in losses or "zinb" in losses
        ) and REGISTRY_KEYS.SIZE_FACTOR_KEY not in self.adata_manager.data_registry:
            raise ValueError(f"Have to register {REGISTRY_KEYS.SIZE_FACTOR_KEY} when using 'nb' or 'zinb' loss.")

        num_groups = 1
        integrate_on_idx = None
        if integrate_on is not None:
            if integrate_on not in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
                raise ValueError(
                    "Cannot integrate on {!r}, has to be one of the registered categorical covariates = {}".format(
                        integrate_on, self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]
                    )
                )
            elif integrate_on in ignore_categories:
                raise ValueError(
                    "Specified integrate_on = {!r} is in ignore_categories = {}.".format(
                        integrate_on, ignore_categories
                    )
                )
            else:
                num_groups = len(
                    self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["mappings"][integrate_on]
                )
                integrate_on_idx = self.adata_manager.registry["setup_args"]["categorical_covariate_keys"].index(
                    integrate_on
                )

        modality_lengths = [adata.uns["modality_lengths"][key] for key in sorted(adata.uns["modality_lengths"].keys())]

        cont_covariate_dims = []
        if len(cont_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)) > 0:
            cont_covariate_dims = [1 for key in cont_covs["columns"] if key not in ignore_categories]

        cat_covariate_dims = []
        if len(cat_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)) > 0:
            cat_covariate_dims = [
                num_cat
                for i, num_cat in enumerate(cat_covs.n_cats_per_key)
                if cat_covs["field_keys"][i] not in ignore_categories
            ]

        self.module = MultiVAETorch(
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
            cat_covariate_dims=cat_covariate_dims,
            cont_covariate_dims=cont_covariate_dims,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            cont_cov_type=cont_cov_type,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_cont_embed=n_hidden_cont_embed,
            mmd=mmd,
            activation=activation,
            initialization=initialization,
        )

        self.init_params_ = self._get_init_params(locals())

    def impute(self, target_modality, adata=None, batch_size=256):
        """Impute values for the target modality."""
        with torch.no_grad():
            self.module.eval()
            if not self.is_trained_:
                raise RuntimeError("Please train the model first.")

            adata = self._validate_anndata(adata)

            scdl = self._make_data_loader(adata=adata, batch_size=batch_size)

            imputed = []
            for tensors in scdl:
                _, generative_outputs = self.module.forward(tensors, compute_loss=False)

                rs = generative_outputs["rs"]
                r = rs[target_modality]
                imputed += [r.cpu()]

            return torch.cat(imputed).squeeze().numpy()

    @torch.inference_mode()
    def get_model_output(self, adata=None, batch_size=256):
        """Return the latent representation."""
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(adata=adata, batch_size=batch_size)

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z_joint"]
            latent += [z.cpu()]

        adata.obsm["X_multiMIL"] = torch.cat(latent).numpy()

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
        **kwargs,
    ):
        """Train the model using amortized variational inference.

        :param max_epochs:
            Number of passes through the dataset
        :param lr:
            Learning rate for optimization
        :param use_gpu:
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False)
        :param train_size:
            Size of training set in the range [0.0, 1.0]
        :param validation_size:
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set
        :param batch_size:
            Minibatch size to use during training
        :param weight_decay:
            Weight decay regularization term for optimization
        :param eps:
            Optimizer eps
        :param early_stopping:
            Whether to perform early stopping with respect to the validation set
        :param save_best:
            Save the best model state with respect to the validation loss, or use the final
            state in the training procedure
        :param check_val_every_n_epoch:
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
            If so, val is checked every epoch
        :param n_epochs_kl_warmup:
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`. Default is 1/3 of `max_epochs`.
        :param n_steps_kl_warmup:
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`
        :param plan_kwargs:
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate
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
            kwargs["callbacks"].append(SaveBestState(monitor="reconstruction_loss_validation"))

        if save_checkpoint_every_n_epochs is not None:
            if path_to_checkpoints is not None:
                kwargs["callbacks"].append(
                    ModelCheckpoint(
                        dirpath=path_to_checkpoints,
                        save_top_k=-1,
                        monitor="epoch",
                        every_n_epochs=save_checkpoint_every_n_epochs,
                        verbose=True,
                    )
                )
            else:
                raise ValueError(
                    f"`save_checkpoint_every_n_epochs` = {save_checkpoint_every_n_epochs} so `path_to_checkpoints` has to be not None but is {path_to_checkpoints}."
                )

        if self.group_column is not None:
            data_splitter = GroupDataSplitter(
                self.adata_manager,
                group_column=self.group_column,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )

        if self.group_column is not None:
            data_splitter = GroupDataSplitter(
                self.adata_manager,
                group_column=self.group_column,
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
            early_stopping_monitor="reconstruction_loss_validation",
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
        """
        if size_factor_key is not None and rna_indices_end is not None:
            raise ValueError(
                "Only one of [`size_factor_key`, `rna_indices_end`] can be specified, but both are not `None`."
            )

        setup_method_args = cls._get_setup_method_args(**locals())

        batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            fields.LayerField(
                REGISTRY_KEYS.X_KEY,
                layer=None,
            ),
            batch_field,
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        # only one can be not None
        if size_factor_key is not None:
            anndata_fields.append(fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key))
        if rna_indices_end is not None:
            if scipy.sparse.issparse(adata.X):
                adata.obs.loc[:, "size_factors"] = adata[:, :rna_indices_end].X.A.sum(1).T.tolist()
            else:
                adata.obs.loc[:, "size_factors"] = adata[:, :rna_indices_end].X.sum(1).T.tolist()
            anndata_fields.append(fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, "size_factors"))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def plot_losses(self, save=None):
        """Plot losses."""
        df = pd.DataFrame(self.history["train_loss_epoch"])
        for key in self.history.keys():
            if key != "train_loss_epoch":
                df = df.join(self.history[key])

        df["epoch"] = df.index

        loss_names = ["kl_local", "elbo", "reconstruction_loss"]
        for i in range(self.module.n_modality):
            loss_names.append(f"modality_{i}_reconstruction_loss")

        if self.module.loss_coefs["integ"] != 0:
            loss_names.append("integ_loss")

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
    # accessed on 5 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: ad.AnnData,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
        freeze: bool = True,
        ignore_categories: Optional[List[str]] = None,
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

        attr_dict, var_names, load_state_dict = _get_loaded_data(reference_model, device=device)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError("Saved model does not contain original setup inputs. " "Cannot load the original setup.")

        if ignore_categories is None:
            ignore_categories = []

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        adata_manager = model.get_anndata_manager(adata, required=True)

        # model tweaking
        num_of_cat_to_add = [
            new_cat - old_cat
            for i, (old_cat, new_cat) in enumerate(
                zip(
                    reference_model.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key,
                    adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key,
                )
            )
            if adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["field_keys"][i] not in ignore_categories
        ]

        model.to_device(device)

        new_state_dict = model.module.state_dict()
        for key, load_ten in load_state_dict.items():  # load_state_dict = old
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                old_shape = new_ten.shape
                new_shape = load_ten.shape
                if old_shape[0] == new_shape[0]:
                    dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                    fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                else:
                    dim_diff = new_ten.size()[0] - load_ten.size()[0]
                    fixed_ten = torch.cat([load_ten, new_ten[-dim_diff:, ...]], dim=0)
                load_state_dict[key] = fixed_ten

        model.module.load_state_dict(load_state_dict)

        # freeze everything but the condition_layer in condMLPs
        if freeze:
            for _, par in model.module.named_parameters():
                par.requires_grad = False
            for i, embed in enumerate(model.module.cat_covariate_embeddings):
                if num_of_cat_to_add[i] > 0:  # unfreeze the ones where categories were added
                    embed.weight.requires_grad = True
            if model.module.integrate_on_idx:
                model.module.theta.requires_grad = True

        model.module.eval()
        model.is_trained_ = False

        return model
