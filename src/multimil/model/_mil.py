import logging
import warnings

import anndata as ad
import torch
from anndata import AnnData
from pytorch_lightning.callbacks import ModelCheckpoint
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base import ArchesMixin, BaseModelClass
from scvi.model.base._archesmixin import _get_loaded_data
from scvi.model.base._utils import _initialize_model
from scvi.train import AdversarialTrainingPlan, TrainRunner
from scvi.train._callbacks import SaveBestState

from multimil.dataloaders import GroupAnnDataLoader, GroupDataSplitter
from multimil.module import MILClassifierTorch
from multimil.utils import (
    get_bag_info,
    get_predictions,
    plt_plot_losses,
    prep_minibatch,
    save_predictions_in_adata,
    select_covariates,
    setup_ordinal_regression,
)

logger = logging.getLogger(__name__)


class MILClassifier(BaseModelClass, ArchesMixin):
    """MultiMIL MIL prediction model.

    Parameters
    ----------
    adata
        AnnData object containing embeddings and covariates.
    sample_key
        Key in `adata.obs` that corresponds to the sample covariate.
    classification
        List of keys in `adata.obs` that correspond to the classification covariates.
    regression
        List of keys in `adata.obs` that correspond to the regression covariates.
    ordinal_regression
        List of keys in `adata.obs` that correspond to the ordinal regression covariates.
    sample_batch_size
        Number of samples per bag, i.e. sample. Default is 128.
    normalization
        One of "layer" or "batch". Default is "layer".
    z_dim
        Dimensionality of the input latent space. Default is 16.
    dropout
        Dropout rate. Default is 0.2.
    scoring
        How to calculate attention scores. One of "gated_attn", "attn", "mean", "max", "sum". Default is "gated_attn".
    attn_dim
        Dimensionality of the hidden layer in attention calculation. Default is 16.
    n_layers_cell_aggregator
        Number of layers in the cell aggregator. Default is 1.
    n_layers_classifier
        Number of layers in the classifier. Default is 2.
    n_layers_regressor
        Number of layers in the regressor. Default is 2.
    n_hidden_cell_aggregator
        Number of hidden units in the cell aggregator. Default is 128.
    n_hidden_classifier
        Number of hidden units in the classifier. Default is 128.
    n_hidden_regressor
        Number of hidden units in the regressor. Default is 128.
    class_loss_coef
        Coefficient for the classification loss. Default is 1.0.
    regression_loss_coef
        Coefficient for the regression loss. Default is 1.0.
    activation
        Activation function. Default is 'leaky_relu'.
    initialization
        Initialization method for the weights. Default is None.
    anneal_class_loss
        Whether to anneal the classification loss. Default is False.

    """

    def __init__(
        self,
        adata,
        sample_key,
        classification=None,
        regression=None,
        ordinal_regression=None,
        sample_batch_size=128,
        normalization="layer",
        dropout=0.2,
        scoring="gated_attn",  # How to calculate attention scores
        attn_dim=16,
        n_layers_cell_aggregator: int = 1,
        n_layers_classifier: int = 2,
        n_layers_regressor: int = 2,
        n_hidden_cell_aggregator: int = 128,
        n_hidden_classifier: int = 128,
        n_hidden_regressor: int = 128,
        class_loss_coef=1.0,
        regression_loss_coef=1.0,
        activation="leaky_relu",  # or tanh
        initialization=None,  # xavier (tanh) or kaiming (leaky_relu)
        anneal_class_loss=False,
    ):
        super().__init__(adata)

        z_dim = adata.X.shape[1]

        if classification is None:
            classification = []
        if regression is None:
            regression = []
        if ordinal_regression is None:
            ordinal_regression = []

        self.sample_key = sample_key
        self.scoring = scoring

        if self.sample_key not in self.adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
            raise ValueError(
                f"Sample key = '{self.sample_key}' has to be one of the registered categorical covariates = {self.adata_manager.registry['setup_args']['categorical_covariate_keys']}"
            )

        if len(classification) + len(regression) + len(ordinal_regression) == 0:
            raise ValueError(
                'At least one of "classification", "regression", "ordinal_regression" has to be specified.'
            )

        self.classification = classification
        self.regression = regression
        self.ordinal_regression = ordinal_regression

        # check if all of the three above were registered with setup anndata
        for key in classification + ordinal_regression:
            if key not in self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["field_keys"]:
                raise ValueError(f"Key '{key}' is not registered as categorical covariates.")

        for key in regression:
            if key not in self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)["field_keys"]:
                raise ValueError(f"Key '{key}' is not registered as continuous covariates.")

        # TODO add check that class is the same within a patient
        # TODO assert length of things is the same as number of modalities
        # TODO add that n_layers has to be > 0 for all
        # TODO warning if n_layers == 1 then n_hidden is not used for classifier
        # TODO check that there is at least on ecovariate to predict

        self.regression_idx = []
        if len(cont_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)) > 0:
            for key in cont_covs["columns"]:
                if key in self.regression:
                    self.regression_idx.append(list(cont_covs["columns"]).index(key))
                # Note: ignore_covariates parameter is kept for API compatibility but not used in MIL-only version

        # classification and ordinal regression together here as ordinal regression values need to be registered as categorical covariates
        self.class_idx, self.ord_idx = [], []
        self.num_classification_classes = []
        if len(cat_covs := self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)) > 0:
            for i, num_cat in enumerate(cat_covs.n_cats_per_key):
                cat_cov_name = cat_covs["field_keys"][i]
                if cat_cov_name in self.classification:
                    self.num_classification_classes.append(num_cat)
                    self.class_idx.append(i)
                elif cat_cov_name in self.ordinal_regression:
                    self.num_classification_classes.append(num_cat)
                    self.ord_idx.append(i)
                # Note: ignore_covariates parameter is kept for API compatibility but not used in MIL-only version

        for label in ordinal_regression:
            print(
                f"The order for {label} ordinal classes is: {adata.obs[label].cat.categories}. If you need to change the order, please rerun setup_anndata and specify the correct order with the `ordinal_regression_order` parameter."
            )

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
            n_layers_regressor=n_layers_regressor,
            n_hidden_regressor=n_hidden_regressor,
            n_hidden_cell_aggregator=n_hidden_cell_aggregator,
            n_hidden_classifier=n_hidden_classifier,
            class_loss_coef=class_loss_coef,
            regression_loss_coef=regression_loss_coef,
            sample_batch_size=sample_batch_size,
            class_idx=self.class_idx,
            ord_idx=self.ord_idx,
            reg_idx=self.regression_idx,
            anneal_class_loss=anneal_class_loss,
        )

        self.init_params_ = self._get_init_params(locals())

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
        adversarial_mixing: bool = False,
        plan_kwargs: dict | None = None,
        early_stopping_monitor: str | None = "accuracy_validation",
        early_stopping_mode: str | None = "max",
        save_checkpoint_every_n_epochs: int | None = None,
        path_to_checkpoints: str | None = None,
        **kwargs,
    ):
        """Trains the model using amortized variational inference.

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
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`. Default is 1/3 of `max_epochs`.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`
        adversarial_mixing
            Whether to use adversarial mixing. Default is False.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        early_stopping_monitor
            Metric to monitor for early stopping. Default is "accuracy_validation".
        early_stopping_mode
            One of "min" or "max". Default is "max".
        save_checkpoint_every_n_epochs
            Save a checkpoint every n epochs.
        path_to_checkpoints
            Path to save checkpoints.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.

        Returns
        -------
        Trainer object.
        """
        if len(self.regression) > 0:
            if early_stopping_monitor == "accuracy_validation":
                warnings.warn(
                    "Setting early_stopping_monitor to 'regression_loss_validation' and early_stopping_mode to 'min' as regression is used.",
                    stacklevel=2,
                )
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
            kwargs["callbacks"].append(SaveBestState(monitor=early_stopping_monitor, mode=early_stopping_mode))

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
        # until here

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
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        ordinal_regression_order: dict[str, list[str]] | None = None,
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
        categorical_covariate_keys
            Keys in `adata.obs` that correspond to categorical data.
        continuous_covariate_keys
            Keys in `adata.obs` that correspond to continuous data.
        ordinal_regression_order
            Dictionary with regression classes as keys and order of classes as values.
        kwargs
            Additional parameters to pass to register_fields() of AnnDataManager.
        """
        if categorical_covariate_keys is not None:
            for key in categorical_covariate_keys:
                adata.obs[key] = adata.obs[key].astype("category")

        setup_ordinal_regression(adata, ordinal_regression_order, categorical_covariate_keys)

        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            fields.LayerField(
                REGISTRY_KEYS.X_KEY,
                layer=None,
            ),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, None),
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_model_output(
        self,
        adata=None,
        batch_size=256,
    ):
        """Save the attention scores and predictions in the adata object.

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

        bag_counter = 0
        cell_counter = 0

        for tensors in scdl:
            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            pred = outputs["predictions"]

            # get attention for each cell in the bag
            if self.scoring in ["gated_attn", "attn"]:
                cell_attn = self.module.cell_level_aggregator[-1].A.squeeze(dim=1)
                cell_attn = cell_attn.flatten()  # in inference always one patient per batch
                cell_level_attn += [cell_attn.cpu()]

            assert outputs["z"].shape[0] % pred[0].shape[0] == 0
            sample_size = outputs["z"].shape[0] // pred[0].shape[0]  # how many cells in patient_minibatch
            minibatch_size, n_samples_in_batch = prep_minibatch(cat_covs, self.module.sample_batch_size)
            regression = select_covariates(cont_covs, self.regression_idx, n_samples_in_batch)
            ordinal_regression = select_covariates(cat_covs, self.ord_idx, n_samples_in_batch)
            classification = select_covariates(cat_covs, self.class_idx, n_samples_in_batch)

            # calculate accuracies of predictions
            bag_class_pred, bag_class_true, class_pred = get_predictions(
                self.class_idx, pred, classification, sample_size, bag_class_pred, bag_class_true, class_pred
            )
            bag_ord_pred, bag_ord_true, ord_pred = get_predictions(
                self.ord_idx,
                pred,
                ordinal_regression,
                sample_size,
                bag_ord_pred,
                bag_ord_true,
                ord_pred,
                len(self.class_idx),
            )
            bag_reg_pred, bag_reg_true, reg_pred = get_predictions(
                self.regression_idx,
                pred,
                regression,
                sample_size,
                bag_reg_pred,
                bag_reg_true,
                reg_pred,
                len(self.class_idx) + len(self.ord_idx),
            )

            # save bag info to be able to calculate bag predictions later
            bags, cell_counter, bag_counter = get_bag_info(
                bags, n_samples_in_batch, minibatch_size, cell_counter, bag_counter, self.module.sample_batch_size
            )

        if self.scoring in ["gated_attn", "attn"]:
            cell_level = torch.cat(cell_level_attn).numpy()
            adata.obs["cell_attn"] = cell_level
        flat_bags = [value for sublist in bags for value in sublist]
        adata.obs["bags"] = flat_bags

        for i in range(len(self.class_idx)):
            name = self.classification[i]
            class_names = self.adata_manager.get_state_registry("extra_categorical_covs")["mappings"][name]
            save_predictions_in_adata(
                adata,
                i,
                self.classification,
                bag_class_pred,
                bag_class_true,
                class_pred,
                class_names,
                name,
                clip="argmax",
            )
        for i in range(len(self.ord_idx)):
            name = self.ordinal_regression[i]
            class_names = self.adata_manager.get_state_registry("extra_categorical_covs")["mappings"][name]
            save_predictions_in_adata(
                adata, i, self.ordinal_regression, bag_ord_pred, bag_ord_true, ord_pred, class_names, name, clip="clip"
            )
        for i in range(len(self.regression_idx)):
            name = self.regression[i]
            reg_names = self.adata_manager.get_state_registry("extra_continuous_covs")["columns"]
            save_predictions_in_adata(
                adata, i, self.regression, bag_reg_pred, bag_reg_true, reg_pred, reg_names, name, clip=None, reg=True
            )

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
    # accessed on 7 November 2022
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: BaseModelClass,
        use_gpu: str | int | bool | None = None,
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

        Returns
        -------
        Model with updated architecture and weights.
        """
        # currently this function works only if the prediction cov is present in the .obs of the query
        # TODO need to allow it to be missing, maybe add a dummy column to .obs of query adata

        _, _, device = parse_use_gpu_arg(use_gpu)

        attr_dict, _, _ = _get_loaded_data(reference_model, device=device)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError("Saved model does not contain original setup inputs. " "Cannot load the original setup.")

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        model.module.load_state_dict(reference_model.module.state_dict())
        model.to_device(device)

        model.module.eval()
        model.is_trained_ = True

        return model
