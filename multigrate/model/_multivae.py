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
from ..train import MultiVAETrainingPlan
from ..distributions import *
from scvi.data._anndata import _setup_anndata
from scvi.dataloaders import DataSplitter, AnnDataLoader
from typing import List, Optional, Union
from scvi.model.base import BaseModelClass
from scvi.train._callbacks import SaveBestState
from scvi.train import TrainRunner
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base._utils import _initialize_model
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

class MultiVAE(BaseModelClass):
    def __init__(
        self,
        adata,
        modality_lengths,
        integrate_on=None,
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
        integrate_on_idx=None,
        cont_cov_type='logsigm',
        n_layers_cont_embed: int = 1,
        n_layers_encoders = [],
        n_layers_decoders = [],
        n_layers_shared_decoder: int = 1,
        n_hidden_cont_embed: int = 32,
        n_hidden_encoders = [],
        n_hidden_decoders = [],
        n_hidden_shared_decoder: int = 32,
        add_shared_decoder = True,
        ignore_categories = [],
    ):

        super().__init__(adata)

        # TODO: add options for number of hidden layers, hidden layers dim and output activation functions
        if normalization not in ['layer', 'batch', None]:
            raise ValueError(f'Normalization has to be one of ["layer", "batch", None]')
        # TODO: do some assertions for other parameters

        num_groups = 1
        integrate_on_idx = None
        if integrate_on:
            if integrate_on not in adata.uns['_scvi']['extra_categoricals']['keys']:
                raise ValueError(f'Cannot integrate on {integrate_on}, has to be one of extra categoricals = {adata.uns["_scvi"]["extra_categoricals"]["keys"]}')
            else:
                num_groups = len(adata.uns['_scvi']['extra_categoricals']['mappings'][integrate_on])
                integrate_on_idx = adata.uns['_scvi']['extra_categoricals']['keys'].index(integrate_on)

        self.adata = adata

        cont_covariate_dims = []
        if adata.uns['_scvi'].get('extra_continuous_keys') is not None:
            cont_covariate_dims = [1 for key in adata.uns['_scvi']['extra_continuous_keys'] if key != 'size_factors' and key not in ignore_categories]

        cat_covariate_dims = []
        if adata.uns['_scvi'].get('extra_categoricals') is not None:
            cat_covariate_dims = [num_cat for i, num_cat in enumerate(adata.uns['_scvi']['extra_categoricals']['n_cats_per_key']) if adata.uns['_scvi']['extra_categoricals']['keys'][i] not in ignore_categories]

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
            integrate_on_idx=integrate_on_idx,
            cat_covariate_dims=cat_covariate_dims,
            cont_covariate_dims=cont_covariate_dims,
            n_layers_encoders=n_layers_encoders,
            n_layers_decoders=n_layers_decoders,
            n_layers_shared_decoder=n_layers_shared_decoder,
            n_hidden_encoders=n_hidden_encoders,
            n_hidden_decoders=n_hidden_decoders,
            n_hidden_shared_decoder=n_hidden_shared_decoder,
            cont_cov_type=cont_cov_type,
            n_layers_cont_embed=n_layers_cont_embed,
            n_hidden_cont_embed=n_hidden_cont_embed,
            add_shared_decoder=add_shared_decoder,
        )

        self.init_params_ = self._get_init_params(locals())

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
        
        adata.obsm['latent'] = torch.cat(latent).numpy()

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
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        n_epochs_kl_warmup = max(max_epochs//3, 1)
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

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = MultiVAETrainingPlan(self.module, **plan_kwargs)
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
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
        )

    # TODO add new losses
    def plot_losses(self):
        df = pd.DataFrame(self.history['train_loss_epoch'])
        for key in self.history.keys():
            if key != 'train_loss_epoch':
                df = df.join(self.history[key])

        df['epoch'] = df.index

        plt.figure(figsize=(15, 10))

        loss_names = ['kl_local', 'elbo', 'reconstruction_loss']
        if self.module.loss_coefs['integ'] != 0:
            loss_names.append('integ')

        nrows = 2

        for i, name in enumerate(loss_names):
            plt.subplot(nrows, 2, i+1)
            plt.plot(df['epoch'], df[name+'_train'], '.-', label=name+'_train')
            plt.plot(df['epoch'], df[name+'_validation'], '.-', label=name+'_validation')
            plt.xlabel('epoch')
            plt.legend()

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
        freeze: bool = True
    ):
        use_gpu, device = parse_use_gpu_arg(use_gpu)

        attr_dict = reference_model._get_user_attributes()
        attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}
        load_state_dict = deepcopy(reference_model.module.state_dict())

        model = _initialize_model(cls, adata, attr_dict)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        # model tweaking
        num_of_cat_to_add = [new_cat - old_cat for old_cat, new_cat in zip(reference_model.adata.uns['_scvi']['extra_categoricals']['n_cats_per_key'], adata.uns['_scvi']['extra_categoricals']['n_cats_per_key'])]
        
        new_state_dict = model.module.state_dict()
        for key, load_ten in load_state_dict.items(): # load_state_dict = old
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

        model.to_device(device)

        # freeze everything but the condition_layer in condMLPs
        if freeze:
            for key, par in model.module.named_parameters():
                par.requires_grad = False
            for i, embed in enumerate(model.module.cat_covariate_embeddings):
                if num_of_cat_to_add[i] > 0: # unfreeze the ones where categories were added
                    embed.weight.requires_grad = True
            if model.module.integrate_on_idx:
                model.module.theta.requires_grad = True

        model.module.eval()
        model.is_trained_ = False

        return model
