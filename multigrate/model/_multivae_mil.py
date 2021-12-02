import torch
import time
import logging
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
from ..dataloaders import BagDataSplitter, BagAnnDataLoader
from scvi.dataloaders import DataSplitter, AnnDataLoader
from typing import List, Optional, Union
from scvi.model.base import BaseModelClass
from scvi.train._callbacks import SaveBestState
from scvi.train import TrainRunner
from ..train import MILTrainingPlan
from scvi.data._anndata import _setup_anndata

logger = logging.getLogger(__name__)

class MultiVAE_MIL(BaseModelClass):
    def __init__(
        self,
        adata,
        modality_lengths,
        class_label,
        patient_label,
        condition_encoders=False,
        condition_decoders=True,
        normalization='layer',
        add_patient_id_to_classifier=False,
        z_dim=15,
        h_dim=32,
        losses=[],
        dropout=0.2,
        cond_dim=10,
        kernel_type='gaussian',
        loss_coefs=[],
        scoring='attn',
        attn_dim=32,
        class_layers=1,
        class_layer_size=128,

        class_loss_coef=1.0,

    ):
        super().__init__(adata)

        self.patient_column = patient_label
        self.scoring = scoring
        self.adata = adata
        num_groups = len(set(self.adata.obs.group))

        if adata.uns['_scvi'].get('extra_continuous_keys') is not None:
            cont_covariate_dims = [1 for key in adata.uns['_scvi']['extra_continuous_keys'] if key != 'size_factors']
        else:
            cont_covariate_dims = []

        if adata.uns['_scvi'].get('extra_categoricals') is not None:
            try:
                cat_covariate_dims = [num_cat for i, num_cat in enumerate(adata.uns['_scvi']['extra_categoricals']['n_cats_per_key']) if adata.uns['_scvi']['extra_categoricals']['keys'][i] != class_label]
                num_classes = len(adata.uns['_scvi']['extra_categoricals']['mappings'][class_label])
            except:
                raise ValueError(f'{class_label} has to be registered as a categorical covariate beforehand with setup_anndata.')
        else:
            raise ValueError('Class labels have to be registered as a categorical covariate beforehand with setup_anndata.')

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
                        class_layers=class_layers,
                        class_layer_size=class_layer_size,
                        class_loss_coef=class_loss_coef
                    )


    def use_model(
        self,
        model,
        freeze_vae=True,
        freeze_cov_embeddings=True
    ):
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

        data_splitter = BagDataSplitter(
            self.adata,
            patient_column=self.patient_column,
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
        class_label,
        rna_indices_end = None,
        categorical_covariate_keys = None,
        continuous_covariate_keys = None
    ):
        if rna_indices_end:
            adata.obs['size_factors'] = adata[:, :rna_indices_end].X.sum(1).T.tolist()[0]

            if continuous_covariate_keys:
                continuous_covariate_keys.append('size_factors')
            else:
                continuous_covariate_keys = ['size_factors']

        if categorical_covariate_keys:
            categorical_covariate_keys.append('group') # from .data._preprocessing.organize_multiome_anndatas
            categorical_covariate_keys.append(class_label) # order important! class label key always last
        else:
            categorical_covariate_keys = ['group', class_label]

        return _setup_anndata(
            adata,
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
        )

    def get_latent_representation(
        self,
        adata=None,
        batch_size=64
    ):
        with torch.no_grad():
            self.module.eval()
            if not self.is_trained_:
                raise RuntimeError("Please train the model first.")

            adata = self._validate_anndata(adata)

            scdl = self._make_data_loader(
                adata=adata,
                batch_size=batch_size,
                min_size_per_class=batch_size, # hack to ensure that not full batches are processed properly
                data_loader_class=BagAnnDataLoader,
                shuffle=False,
                shuffle_classes=False,
                patient_column=self.patient_column,
                drop_last=False,
            )

            latent, cell_level_attn, cov_level_attn = [], [], []
            for tensors in scdl:
                inference_inputs = self.module._get_inference_input(tensors)
                outputs = self.module.inference(**inference_inputs)
                z = outputs['z_joint']
                cell_attn = self.module.cell_level_aggregator[1].A.squeeze()
                cell_attn = cell_attn.flatten()
                cov_attn = self.module.classifier[1].A.squeeze()
                cov_attn = cov_attn.flatten()
                cov_attn = cov_attn.unsqueeze(0).repeat(z.shape[0], 1)

                latent += [z.cpu()]
                cell_level_attn += [cell_attn.cpu()]
                cov_level_attn += [cov_attn.cpu()]
            return torch.cat(latent).numpy(), torch.cat(cell_level_attn).numpy(), torch.cat(cov_level_attn).numpy()
