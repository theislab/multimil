import sys
import torch
import time
import os

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import logging
from torch import nn
from anndata import AnnData
from copy import deepcopy
from collections import defaultdict, Counter
from operator import itemgetter, attrgetter
from torch.nn import functional as F
from itertools import cycle, zip_longest, groupby
from ..module import MultiVAETorch
from ..distributions import *
from scvi.data._anndata import _setup_anndata
from scvi.dataloaders import DataSplitter, AnnDataLoader
from typing import List, Optional, Union
from scvi.model.base import BaseModelClass
from scvi.train._callbacks import SaveBestState
from scvi.train import AdversarialTrainingPlan, TrainRunner
from scvi.model._utils import parse_use_gpu_arg

class MultiVAE(BaseModelClass):
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
        loss_coefs=[]
    ):

        super().__init__(adata)

        # TODO: add options for number of hidden layers, hidden layers dim and output activation functions

        if normalization not in ['layer', 'batch', None]:
            raise ValueError(f'Normalization has to be one of ["layer", "batch", None]')
        # TODO: do some assertions for other parameters

        self.adata = adata
        num_groups = len(set(self.adata.obs.group))

        cat_covariate_dims = [num_cat for i, num_cat in enumerate(adata.uns['_scvi']['extra_categoricals']['n_cats_per_key'])]
        cont_covariate_dims = [1 for key in adata.uns['_scvi']['extra_continuous_keys'] if key != 'size_factors']

        self.module = MultiVAETorch(
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
            cat_covariate_dims=cat_covariate_dims,
            cont_covariate_dims=cont_covariate_dims,
        )

        # self.init_params_ = self._get_init_params(locals())

    def impute(
        self,
        target_modality,
        adata=None,
        batch_size=64
    ):
        with torch.no_grad():
            self.module.eval()
            if not self.is_trained_:
                raise RuntimeError("Please train the model first.")

            adata = self._validate_anndata(adata)

            scdl = self._make_data_loader(
                adata=self.adata, batch_size=batch_size
            )

            imputed = []
            for tensors in scdl:
                _, generative_outputs = self.module.forward(
                    tensors,
                    compute_loss=False
                )

                rs = generative_outputs['rs']
                r = rs[target_modality]
                imputed += [r.cpu()]

            return torch.cat(imputed).squeeze().numpy()

    # TODO fix to work with  @torch.no_grad()
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
                adata=adata, batch_size=batch_size
            )

            latent = []
            for tensors in scdl:
                inference_inputs = self.module._get_inference_input(tensors)
                outputs = self.module.inference(**inference_inputs)
                z = outputs['z_joint']
                latent += [z.cpu()]
            return torch.cat(latent).numpy()

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

        if categorical_covariate_keys:
            categorical_covariate_keys.append('group') # from .data._preprocessing.organize_multiome_anndatas
        else:
            categorical_covariate_keys = ['group']
        return _setup_anndata(
            adata,
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
        )

    # TODO
    def plot_losses(
        self,
        recon=True,
        kl=True,
        integ=True,
        cycle=False
    ):
        pass

    def load_query_data(
        adata: AnnData,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
        freeze: bool = True
    ):
        use_gpu, device = parse_use_gpu_arg(use_gpu)

        model = deepcopy(reference_model)
        model.adata = adata

        # check if the reference model was conditioned and update cond_embedding
        n_new_batches = len(set(reference_model.adata.obs.group))
        if reference_model.module.cond_embedding:
            n_batches, cond_dim = reference_model.module.cond_embedding.weight.shape
            old_embed = reference_model.module.cond_embedding.weight.data
            model.module.cond_embedding = nn.Embedding(n_batches+n_new_batches, cond_dim)
            model.module.cond_embedding.weight.data[:n_batches, :] = old_embed
        else:
            raise NotImplementedError('The reference model has to be conditioned.')

        # add another dim to theta if ref model has it
        if reference_model.module.theta is not None:
            rna_length, n_batches = reference_model.module.theta.shape
            old_theta = reference_model.module.theta.data
            model.module.theta = torch.nn.Parameter(torch.randn(rna_length, n_batches+n_new_batches))
            model.module.theta.data[:, :n_batches] = old_theta

        model.to_device(device)

        # freeze everything but the condition_layer in condMLPs
        if freeze:
            for key, par in model.module.named_parameters():
                if not any(module in key for module in ['theta', 'cond_embedding', 'condition_layer']):
                    par.requires_grad = False

        model.module.eval()
        model.is_trained_ = False
        return model
