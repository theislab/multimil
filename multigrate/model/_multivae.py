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
from collections import defaultdict, Counter
from operator import itemgetter, attrgetter
from torch.nn import functional as F
from itertools import cycle, zip_longest, groupby
from ..nn import *
from ..module import MultiVAETorch
from ..distributions import *
from scvi.data._anndata import _setup_anndata
from scvi.dataloaders import DataSplitter, AnnDataLoader
from typing import List, Optional, Union
from scvi.model.base import BaseModelClass, ArchesMixin
from scvi.train._callbacks import SaveBestState
from scvi.train import AdversarialTrainingPlan, TrainRunner

class MultiVAE(BaseModelClass, ArchesMixin):
    def __init__(
        self,
        adata,
        modality_lengths=[],
        condition_encoders=False,
        condition_decoders=True,
        normalization='layer',
        z_dim=15,
        h_dim=32,
        hiddens=[],
        losses=[],
        output_activations=[],
        shared_hiddens=[],
        dropout=0.2,
        device=None,
        theta=None,
        cond_dim=10,
        kernel_type='not gaussian',
        loss_coefs=[]
    ):

        super().__init__(adata)
        # configure to CUDA if is available
        # TODO work with scvi move data
        device =  device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if normalization not in ['layer', 'batch', None]:
            raise ValueError(f'Normalization has to be one of ["layer", "batch", None]')
        # TODO: do some assertions for other parameters

        self.adata = adata
        self.condition_encoders = condition_encoders
        self.condition_decoders = condition_decoders
        self.hiddens = hiddens
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.shared_hiddens = shared_hiddens
        self.dropout = dropout
        self.output_activations = output_activations
        self.input_dims = modality_lengths if len(modality_lengths) > 0 else [len(self.adata.var_names)]
        self.normalization = normalization # need for architecture surgery
        self.n_modality = len(self.input_dims)

        if self.n_modality != len(hiddens):
            if len(hiddens) == 0:
                hiddens = [[] for _ in range(self.n_modality)]
            else:
                raise ValueError(f'hiddens must be the same length as the number of modalities. n_modalities = {self.n_modality} != {len(hiddens)} = len(hiddens)')

        if self.n_modality != len(output_activations):
            if len(output_activations) == 0:
                output_activations = ['linear' for _ in range(self.n_modality)] # or leaky relu?
            else:
                raise ValueError(f'output_activations must be the same length as the number of modalities. n_modalities = {self.n_modality} != {len(output_activations)} = len(output_activations)')

        # TODO fix
        if len(losses) == 0:
            self.losses = ['mse']*self.n_modality
        elif len(losses) == self.n_modality:
            self.losses = losses
        else:
            raise ValueError(f'adatas and losses arguments must be the same length or losses has to be []. len(adatas) = {len(adatas)} != {len(losses)} = len(losses)')

        self.loss_coefs = {'recon': 1,
                          'kl': 1e-6,
                          'integ': 0,
                          'cycle': 0,
                          'nb': 1,
                          'zinb': 1,
                          'mse': 1,
                          'bce': 1}
        self.loss_coefs.update(loss_coefs)

        if self.condition_encoders or self.condition_decoders:
            num_groups = len(set(self.adata.obs.group))
            self.cond_embedding = torch.nn.Embedding(num_groups, cond_dim)
        else:
            self.cond_embedding = None
            cond_dim = 0 # not to add extra input dimentions later to the encoders

        # assume for now that can only use nb/zinb once, i.e. for RNA-seq modality
        # TODO: add check for multiple nb/zinb losses given
        self.theta = theta
        if not self.theta:
            for i, loss in enumerate(losses):
                if loss in ["nb", "zinb"]:
                    groups = list(self.adata.obs.group)
                    self.theta = torch.nn.Parameter(torch.randn(self.input_dims[i], max(len(set(self.adata.obs.group)), 1)))#.to(device).detach().requires_grad_(True)
                    break

        # need for surgery TODO check
        # self.mod_dec_dim = h_dim
        # create modules
        self.shared_decoder = CondMLP(z_dim + self.n_modality, h_dim, dropout_rate=dropout, normalization=normalization)
        cond_dim_enc = cond_dim if self.condition_encoders else 0
        self.encoders = [CondMLP(x_dim, z_dim, embed_dim=cond_dim_enc, dropout_rate=dropout, normalization=normalization) for x_dim in self.input_dims]
        cond_dim_dec = cond_dim if self.condition_decoders else 0
        self.decoders = [Decoder(h_dim, x_dim, embed_dim=cond_dim_dec, dropout_rate=dropout, normalization=normalization, loss=loss) for x_dim, loss in zip(self.input_dims, self.losses)]


        self.mus = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]
        self.logvars = [nn.Linear(z_dim, z_dim) for _ in self.input_dims]

        self.module = MultiVAETorch(self.encoders, self.decoders, self.shared_decoder,
                                   self.mus, self.logvars, self.theta,
                                   device, self.condition_encoders, self.condition_decoders, self.cond_embedding,
                                   self.input_dims, self.losses, self.loss_coefs, kernel_type)

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
        rna_indices_end = None
    ):
        if rna_indices_end:
            adata.obs['size_factors'] = adata[:, :rna_indices_end].X.sum(1).T.tolist()[0]
            continuous_covariate_keys = ['size_factors']
        else:
            continuous_covariate_keys = None

        return _setup_anndata(
            adata,
            batch_key='group', # not a real batch, but batches
            continuous_covariate_keys=continuous_covariate_keys
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
