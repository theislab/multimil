import logging
from typing import Literal

import anndata as ad
import torch
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

from multimil.dataloaders import GroupDataSplitter
from multimil.module import MultiVAETorch
from multimil.utils import calculate_size_factor, plt_plot_losses

logger = logging.getLogger(__name__)


class MultiVAE(BaseModelClass, ArchesMixin):
    """MultiMIL multimodal integration model.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~multigrate.model.MultiVAE.setup_anndata`.
    integrate_on
        One of the categorical covariates refistered with :math:`~multigrate.model.MultiVAE.setup_anndata` to integrate on. The latent space then will be disentangled from this covariate. If `None`, no integration is performed.
    condition_encoders
        Whether to concatentate covariate embeddings to the first layer of the encoders. Default is `False`.
    condition_decoders
        Whether to concatentate covariate embeddings to the first layer of the decoders. Default is `True`.
    normalization
        What normalization to use; has to be one of `batch` or `layer`. Default is `layer`.
    z_dim
        Dimensionality of the latent space. Default is 16.
    losses
        Which losses to use for each modality. Has to be the same length as the number of modalities. Default is `MSE` for all modalities.
    dropout
        Dropout rate. Default is 0.2.
    cond_dim
        Dimensionality of the covariate embeddings. Default is 16.
    kernel_type
        Type of kernel to use for the MMD loss. Default is `gaussian`.
    loss_coefs
        Loss coeficients for the different losses in the model. Default is 1 for all.
    cont_cov_type
        How to calculate embeddings for continuous covariates. Default is `logsim`.
    n_layers_cont_embed
        Number of layers for the continuous covariate embedding calculation. Default is 1.
    n_layers_encoders
        Number of layers for each encoder. Default is 2 for all modalities. Has to be the same length as the number of modalities.
    n_layers_decoders
        Number of layers for each decoder. Default is 2 for all modalities. Has to be the same length as the number of modalities.
    n_hidden_cont_embed
        Number of nodes for each hidden layer in the continuous covariate embedding calculation. Default is 32.
    n_hidden_encoders
        Number of nodes for each hidden layer in the encoders. Default is 32.
    n_hidden_decoders
        Number of nodes for each hidden layer in the decoders. Default is 32.
    mmd
        Which MMD loss to use. Default is `latent`.
    activation
        Activation function to use. Default is `leaky_relu`.
    initialization
        Initialization method to use. Default is `None`.
    ignore_covariates
        List of covariates to ignore. Needed for query-to-reference mapping. Default is `None`.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        integrate_on: str | None = None,
        condition_encoders: bool = True,
        condition_decoders: bool = True,
        normalization: Literal["layer", "batch", None] = "layer",
        z_dim: int = 16,
        losses: list[str] | None = None,
        dropout: float = 0.2,
        cond_dim: int = 16,
        kernel_type: Literal["gaussian", None] = "gaussian",
        loss_coefs: dict[int, float] = None,
        cont_cov_type: Literal["logsim", "sigm", None] = "logsigm",
        n_layers_cont_embed: int = 1,  # TODO default to None?
        n_layers_encoders: list[int] | None = None,
        n_layers_decoders: list[int] | None = None,
        n_hidden_cont_embed: int = 32,  # TODO default to None?
        n_hidden_encoders: list[int] | None = None,
        n_hidden_decoders: list[int] | None = None,
        mmd: Literal["latent", "marginal", "both"] = "latent",
        activation: str | None = "leaky_relu",  # TODO add which options are impelemted
        initialization: str | None = None,  # TODO add which options are impelemted
        ignore_covariates: list[str] | None = None,
    ):
        super().__init__(adata)

        # for the integration with MMD loss
        self.group_column = integrate_on

        # TODO: add options for number of hidden layers, hidden layers dim and output activation functions
        if normalization not in ["layer", "batch", None]:
            raise ValueError('Normalization has to be one of ["layer", "batch", None]')
        # TODO: do some assertions for other parameters

        if ignore_covariates is None:
            ignore_covariates = []

        if (
            "nb" in losses or "zinb" in losses
        ) and REGISTRY_KEYS.SIZE_FACTOR_KEY not in self.adata_manager.data_registry:
            raise ValueError(f"Have to register {REGISTRY_KEYS.SIZE_FACTOR_KEY} when using 'nb' or 'zinb' loss.")

        self.num_groups = 1
        self.integrate_on_idx = None
        if integrate_on is not None:
            if integrate_on not in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
                raise ValueError(
                    "Cannot integrate on {!r}, has to be one of the registered categorical covariates = {}".format(
                        integrate_on, self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]
                    )
                )
            elif integrate_on in ignore_covariates:
                raise ValueError(
                    f"Specified integrate_on = {integrate_on!r} is in ignore_covariates = {ignore_covariates}."
                )
            else:
                self.num_groups = len(
                    self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["mappings"][integrate_on]
                )
                self.integrate_on_idx = self.adata_manager.registry["setup_args"]["categorical_covariate_keys"].index(
                    integrate_on
                )

        self.modality_lengths = [
            adata.uns["modality_lengths"][key] for key in sorted(adata.uns["modality_lengths"].keys())
        ]

        self.cont_covs_idx = []
        self.cont_covariate_dims = []
        if len(cont_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)) > 0:
            for i, key in enumerate(cont_covs["columns"]):
                if key not in ignore_covariates:
                    self.cont_covs_idx.append(i)
                    self.cont_covariate_dims.append(1)

        self.cat_covs_idx = []
        self.cat_covariate_dims = []
        if len(cat_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)) > 0:
            for i, num_cat in enumerate(cat_covs.n_cats_per_key):
                if cat_covs["field_keys"][i] not in ignore_covariates:
                    self.cat_covs_idx.append(i)
                    self.cat_covariate_dims.append(num_cat)

        self.cat_covs_idx = torch.tensor(self.cat_covs_idx)
        self.cont_covs_idx = torch.tensor(self.cont_covs_idx)

        self.module = MultiVAETorch(
            modality_lengths=self.modality_lengths,
            condition_encoders=condition_encoders,
            condition_decoders=condition_decoders,
            normalization=normalization,
            z_dim=z_dim,
            losses=losses,
            dropout=dropout,
            cond_dim=cond_dim,
            kernel_type=kernel_type,
            loss_coefs=loss_coefs,
            num_groups=self.num_groups,
            integrate_on_idx=self.integrate_on_idx,
            cat_covs_idx=self.cat_covs_idx,
            cont_covs_idx=self.cont_covs_idx,
            cat_covariate_dims=self.cat_covariate_dims,
            cont_covariate_dims=self.cont_covariate_dims,
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

    @torch.inference_mode()
    def impute(self, adata=None, batch_size=256):
        """Impute missing values in the adata object.

        Parameters
        ----------
        adata
            AnnData object to run the model on. If `None`, the model's AnnData object is used.
        batch_size
            Minibatch size to use. Default is 256.
        """
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(adata=adata, batch_size=batch_size)

        imputed = [[] for _ in range(len(self.modality_lengths))]

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
            outputs = self.module.generative(**generative_inputs)
            for i, output in enumerate(outputs["rs"]):
                imputed[i] += [output.cpu()]
        for i in range(len(imputed)):
            imputed[i] = torch.cat(imputed[i]).numpy()
            adata.obsm[f"imputed_modality_{i}"] = imputed[i]

    @torch.inference_mode()
    def get_model_output(self, adata=None, batch_size=256):
        """Save the latent representation in the adata object.

        Parameters
        ----------
        adata
            AnnData object to run the model on. If `None`, the model's AnnData object is used.
        batch_size
            Minibatch size to use. Default is 256.
        """
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
        use_gpu: str | int | bool | None = None,
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        n_steps_kl_warmup: int | None = None,
        adversarial_mixing: bool = False,  # TODO check if suppored by us, i don't think it is
        plan_kwargs: dict | None = None,
        save_checkpoint_every_n_epochs: int | None = None,
        path_to_checkpoints: str | None = None,
        **kwargs,
    ):
        """Train the model using amortized variational inference.

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
            Weight decay regularization term for optimization.
        eps
            Optimizer eps.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            Save the best model state with respect to the validation loss, or use the final
            state in the training procedure.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
            If so, val is checked every epoch.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`. Default is 1/3 of `max_epochs`.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        adversarial_mixing
            Whether to use adversarial mixing. Default is `False`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        save_checkpoint_every_n_epochs
            Save a checkpoint every n epochs. If `None`, no checkpoints are saved.
        path_to_checkpoints
            Path to save checkpoints. Required if `save_checkpoint_every_n_epochs` is not `None`.
        kwargs
            Additional keyword arguments for :class:`~scvi.train.TrainRunner`.

        Returns
        -------
        Trainer object.
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
        size_factor_key: str | None = None,
        rna_indices_end: int | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """Set up :class:`~anndata.AnnData` object.

        A mapping will be created between data fields used by ``scvi`` to their respective locations in adata.
        This method will also compute the log mean and log variance per batch for the library size prior.
        None of the data in adata are modified. Only adds fields to adata.

        Parameters
        ----------
        adata
            AnnData object containing raw counts. Rows represent cells, columns represent features.
        size_factor_key
            Key in `adata.obs` containing the size factor. If `None`, will be calculated from the RNA counts.
        rna_indices_end
            Integer to indicate where RNA feature end in the AnnData object. Is used to calculate ``libary_size``.
        categorical_covariate_keys
            Keys in `adata.obs` that correspond to categorical data.
        continuous_covariate_keys
            Keys in `adata.obs` that correspond to continuous data.
        kwargs
            Additional parameters to pass to register_fields() of AnnDataManager.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            fields.LayerField(
                REGISTRY_KEYS.X_KEY,
                layer=None,
            ),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, None),  # TODO check if need to add if it's None
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        size_factor_key = calculate_size_factor(adata, size_factor_key, rna_indices_end)
        anndata_fields.append(fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def plot_losses(self, save=None):
        """Plot losses.

        Parameters
        ----------
        save
            If not None, save the plot to this location.
        """
        loss_names = self.module.select_losses_to_plot()
        plt_plot_losses(self.history, loss_names, save)

    # adjusted from scvi-tools
    # https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/model/base/_archesmixin.py#L30
    # accessed on 5 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: ad.AnnData,
        reference_model: BaseModelClass,
        use_gpu: str | int | bool | None = None,
        freeze: bool = True,
        ignore_covariates: list[str] | None = None,
    ) -> BaseModelClass:
        """Online update of a reference model with scArches algorithm # TODO cite.

        Parameters
        ----------
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        reference_model
            Already instantiated model of the same class.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).
        freeze
            Whether to freeze the encoders and decoders and only train the new weights.
        ignore_covariates
            List of covariates to ignore. Needed for query-to-reference mapping. Default is `None`.

        Returns
        -------
        Model with updated architecture and weights.
        """
        _, _, device = parse_use_gpu_arg(use_gpu)

        attr_dict, _, load_state_dict = _get_loaded_data(reference_model, device=device)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError("Saved model does not contain original setup inputs. " "Cannot load the original setup.")

        if ignore_covariates is None:
            ignore_covariates = []

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        adata_manager = model.get_anndata_manager(adata, required=True)

        # TODO add an exception if need to add new categories but condition_encoders is False
        # model tweaking
        num_of_cat_to_add = [
            new_cat - old_cat
            for i, (old_cat, new_cat) in enumerate(
                zip(
                    reference_model.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key,
                    adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key,
                    strict=False,
                )
            )
            if adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["field_keys"][i] not in ignore_covariates
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

        # freeze everything but the embeddings that have new categories
        if freeze is True:
            for _, par in model.module.named_parameters():
                par.requires_grad = False
            for i, embed in enumerate(model.module.cat_covariate_embeddings):
                if num_of_cat_to_add[i] > 0:  # unfreeze the ones where categories were added
                    embed.weight.requires_grad = True
            if model.module.integrate_on_idx is not None:
                model.module.theta.requires_grad = True

        model.module.eval()
        model.is_trained_ = False

        return model
