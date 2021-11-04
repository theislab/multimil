import torch
import time
from torch import nn
from torch.nn import functional as F
import numpy as np
import scanpy as sc
from operator import attrgetter
from itertools import cycle, zip_longest, groupby
from scipy import spatial
from ..nn import *
from ._multivae import MultiVAE
from ..module import MultiVAETorch_MIL
from scvi.dataloaders import DataSplitter, AnnDataLoader
from typing import List, Optional, Union
from scvi.model.base import BaseModelClass
from scvi.train._callbacks import SaveBestState
from scvi.train import AdversarialTrainingPlan, TrainRunner
from scvi.data._anndata import _setup_anndata

class MultiVAE_MIL(BaseModelClass):
    def __init__(
        self,
        adata,
        modality_lengths,
        condition_encoders=False,
        condition_decoders=True,
        normalization='layer',
        z_dim=15,
        h_dim=32,
        losses=[],
        dropout=0.2,
        cond_dim=10,
        kernel_type='not gaussian',
        loss_coefs=[],
        # mil specific
        class_columns=None,
        bag_key=None,
        scoring='attn',
        attn_dim=32,
        covariate_embed_dim=10,
        class_layers=1,
        class_layer_size=128,
        class_category='condition',
        class_loss_coef=1.0
    ):
        super().__init__(adata)

        self.bag_key = bag_key
        self.class_columns = class_columns
        self.scoring = scoring
        self.adata = adata
        num_groups = len(set(self.adata.obs.group))

        cat_covariate_dims = [num_cat for i, num_cat in enumerate(adata.uns['_scvi']['extra_categoricals']['n_cats_per_key']) if adata.uns['_scvi']['extra_categoricals']['keys'][i] != class_category]
        cont_covariate_dims = [1 for key in adata.uns['_scvi']['extra_continuous_keys'] if key != 'size_factors']
        num_classes = len(adata.uns['_scvi']['extra_categoricals']['mappings'][class_category])

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
                        # mil specific
                        num_classes=num_classes,
                        scoring=scoring,
                        attn_dim=attn_dim,
                        cat_covariate_dims=cat_covariate_dims,
                        cont_covariate_dims=cont_covariate_dims,
                        covariate_embed_dim=covariate_embed_dim,
                        class_layers=class_layers,
                        class_layer_size=class_layer_size,
                        class_loss_coef=class_loss_coef
                    )


    def use_model(self, model, freeze=True):
        self.model = MultiVAETorch_MIL(model.encoders, model.decoders, model.shared_encoder, model.shared_decoder,
                                   model.mu, model.logvar, model.modality_vecs, model.device, model.condition, model.n_batch_labels,
                                   model.pair_groups_dict, model.modalities_per_group, model.paired_networks_per_modality_pairs,
                                   self.num_classes, self.scoring, self.classifier_hiddens, self.normalization, self.dropout)
        if freeze:
            for param in self.model.named_parameters():
                if not param[0].startswith('classifier'):
                    param[1].requires_grad = False

    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        n_epochs_kl_warmup: Optional[int] = 50,
        adversarial_mixing: bool = True,
        plan_kwargs: Optional[dict] = None,
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
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        update_dict = dict(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            n_steps_kl_warmup=n_steps_kl_warmup,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping=early_stopping,
            early_stopping_monitor="reconstruction_loss_validation",
            early_stopping_patience=50,
            optimizer="AdamW",
            scale_adversarial_loss=1
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

        data_splitter = DataSplitter(
            self.adata,
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
            **kwargs,
        )
        return runner()

    # TODO
    def save(self, path):
        torch.save({
            'state_dict' : self.module.state_dict(),
        }, os.path.join(path, 'last-model.pt'), pickle_protocol=4)
        pd.DataFrame(self._val_history).to_csv(os.path.join(path, 'history.csv'))

    # TODO
    def load(self, path):
        model_file = torch.load(os.path.join(path, 'last-model.pt'), map_location=self.device)
        self.module.load_state_dict(model_file['state_dict'])
        self._val_history = pd.read_csv(os.path.join(path, 'history.csv'), index_col=0)

    def setup_anndata(
        adata,
        rna_indices_end = None,
        categorical_covariate_keys = None,
        continuous_covariate_keys = None,
    ):
        if rna_indices_end:
            adata.obs['size_factors'] = adata[:, :rna_indices_end].X.sum(1).T.tolist()[0]

            if continuous_covariate_keys:
                continuous_covariate_keys.append('size_factors')
            else:
                continuous_covariate_keys = ['size_factors']

        return _setup_anndata(
            adata,
            batch_key='group',
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
        )
