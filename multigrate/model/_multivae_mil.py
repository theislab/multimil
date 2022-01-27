import torch
import scvi
import time
import logging
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
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
        class_label,
        patient_label,
        patient_batch_size=128,
        integrate_on=None,
        condition_encoders=False,
        condition_decoders=True,
        normalization='layer',
        n_layers_encoders = [],
        n_layers_decoders = [],
        n_layers_shared_decoder: int = 1,
        n_hidden_encoders = [],
        n_hidden_decoders = [],
        n_hidden_shared_decoder: int = 32,
        add_patient_to_classifier=False,
        hierarchical_attn=True,
        add_shared_decoder=True,
        z_dim=16,
        h_dim=32,
        losses=[],
        dropout=0.2,
        cond_dim=16,
        kernel_type='gaussian',
        loss_coefs=[],
        scoring='gated_attn',
        attn_dim=16,
        n_layers_cell_aggregator: int = 1,
        n_layers_cov_aggregator: int = 1,
        n_layers_classifier: int = 1,
        n_layers_mlp_attn = None,
        n_layers_cont_embed: int = 1,
        n_hidden_cell_aggregator: int = 16,
        n_hidden_cov_aggregator: int = 16,
        n_hidden_classifier: int = 16,
        n_hidden_cont_embed: int = 16,
        n_hidden_mlp_attn = None,
        attention_dropout=True,
        class_loss_coef=1.0,
        reg_coef=1,
        regularize_cell_attn=False,
        regularize_cov_attn=False,
        regularize_vae=False,
        cont_cov_type='logsigm',
    ):
        super().__init__(adata)

        self.patient_column = patient_label
        patient_idx = adata.uns['_scvi']['extra_categoricals']['keys'].index(patient_label)
        self.scoring = scoring
        self.adata = adata
        self.hierarchical_attn = hierarchical_attn
        self.class_label = class_label

        # TODO add check that class is the same within a patient
        # TODO assert length of things is the same as number of modalities
        # TODO add that n_layers has to be > 0 for all
        # TODO warning if n_layers == 1 then n_hidden is not used for classifier and MLP attention
        # TODO warning if MLP attention is used but n layers and n hidden not given that using default values
        if scoring == 'MLP':
            if not n_layers_mlp_attn:
                n_layers_mlp_attn = 1
            if not n_hidden_mlp_attn:
                n_hidden_mlp_attn = 16

        cont_covariate_dims = []
        if adata.uns['_scvi'].get('extra_continuous_keys') is not None:
            cont_covariate_dims = [1 for key in adata.uns['_scvi']['extra_continuous_keys'] if key != 'size_factors']

        num_groups = 1
        integrate_on_idx = None
        if integrate_on:
            try:
                num_groups = len(adata.uns['_scvi']['extra_categoricals']['mappings'][integrate_on])
                integrate_on_idx = adata.uns['_scvi']['extra_categoricals']['keys'].index(integrate_on)
            except:
                raise ValueError(f'Cannot integrate on {integrate_on}, has to be one of extra categoricals = {adata.uns["_scvi"]["extra_categoricals"]["keys"]}')

        if adata.uns['_scvi'].get('extra_categoricals') is not None:
            cat_covariate_dims = [num_cat for i, num_cat in enumerate(adata.uns['_scvi']['extra_categoricals']['n_cats_per_key']) if adata.uns['_scvi']['extra_categoricals']['keys'][i] != class_label]
            try:
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
                        n_hidden_cell_aggregator=n_hidden_cell_aggregator,
                        n_hidden_cov_aggregator=n_hidden_cov_aggregator,
                        n_hidden_classifier=n_hidden_classifier,
                        n_hidden_mlp_attn=n_hidden_mlp_attn,
                        class_loss_coef=class_loss_coef,
                        reg_coef=reg_coef,
                        add_patient_to_classifier=add_patient_to_classifier,
                        patient_idx=patient_idx,
                        hierarchical_attn=hierarchical_attn,
                        patient_batch_size=patient_batch_size,
                        regularize_cell_attn=regularize_cell_attn,
                        regularize_cov_attn=regularize_cov_attn,
                        regularize_vae=regularize_vae,
                        attention_dropout=attention_dropout,
                    )
                    
        self.init_params_ = self._get_init_params(locals())

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
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        n_steps_kl_warmup: Optional[int] = None,
        #n_epochs_kl_warmup: Optional[int] = 50,
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
            categorical_covariate_keys.append(class_label) # order important! class label key always last
        else:
            categorical_covariate_keys = [class_label]

        return _setup_anndata(
            adata,
            continuous_covariate_keys=continuous_covariate_keys,
            categorical_covariate_keys=categorical_covariate_keys,
        )

    def setup_query(
        adata,
        query,
        class_label=None,
        rna_indices_end=None, 
        categorical_covariate_keys = None,
        continuous_covariate_keys = None,
    ):
        MultiVAE.setup_anndata(
            query,
            rna_indices_end=rna_indices_end, 
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
        )
        scvi.data.transfer_anndata_setup(
            adata,
            query,
            extend_categories = True,
        )

    @auto_move_data
    def get_model_output(
        self,
        adata=None,
        batch_size=256,
        inplace=True,
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
            latent, cell_level_attn, cov_level_attn, predictions = [], [], [], []
            bag_true, bag_pred = [], []
            for tensors in scdl:
                class_label = tensors.get(_CONSTANTS.CAT_COVS_KEY)[:, -1]
                batch_size = class_label.shape[0]
                idx = list(range(self.module.patient_batch_size,  batch_size, self.module.patient_batch_size)) # or depending on model.train() and model.eval() ???
                if batch_size % self.module.patient_batch_size != 0: # can only happen during inference for last batches for each patient
                    idx = []

                class_label = torch.tensor_split(class_label, idx, dim=0)
                class_label = [torch.Tensor([labels[0]]).long().to(self.device) for labels in class_label]
                class_label = torch.cat(class_label, dim=0)
                inference_inputs = self.module._get_inference_input(tensors)
                outputs = self.module.inference(**inference_inputs)
                z = outputs['z_joint']
                pred = outputs['prediction']
                cell_attn = self.module.cell_level_aggregator[-1].A.squeeze(dim=1)
                size = cell_attn.shape[-1]
                cell_attn = cell_attn.flatten() # in inference always one patient per batch

                if self.hierarchical_attn:
                    cov_attn = self.module.cov_level_aggregator[-1].A.squeeze(dim=1) # aggregator is always last after hidden MLP layers 
                    cov_attn = cov_attn.unsqueeze(0).repeat(size, 1, 1)
                    cov_attn = cov_attn.flatten(start_dim=0, end_dim=1)
                    cov_level_attn += [cov_attn.cpu()]

                bag_pred += [torch.argmax(pred, dim=-1).cpu()]
                bag_true += [class_label.cpu()]
                pred = torch.argmax(pred, dim=-1).unsqueeze(0).repeat(size, 1)
                pred = pred.flatten()
                predictions += [pred.cpu()]
                latent += [z.cpu()]
                cell_level_attn += [cell_attn.cpu()]
                
            if len(cov_level_attn) == 0:
                cov_level_attn = [torch.Tensor()]

            latent = torch.cat(latent).numpy()
            cell_level = torch.cat(cell_level_attn).numpy()
            cov_level = torch.cat(cov_level_attn).numpy()
            prediction = torch.cat(predictions).numpy()
            bag_pred = torch.cat(bag_pred).numpy()
            bag_true = torch.cat(bag_true).numpy()

            if inplace:
                adata.obsm['latent'] = latent
                adata.obsm['cov_attn'] = cov_level
                adata.obs['cell_attn'] = cell_level
                adata.obs['predicted_class'] = prediction
                adata.uns['bag_true'] = bag_true
                adata.uns['bag_pred'] = bag_pred
            else:
                return latent, cell_level, cov_level, prediction, bag_true, bag_pred

    def classification_report(self, adata=None, level='patient'):
        
        adata = self._validate_anndata(adata)
        if 'predicted_class' not in adata.obs.keys():
            raise RuntimeError(f'"predicted_class" not in adata.obs.keys(), please run model.get_model_output(adata) first.')

        target_names = adata.uns['_scvi']['extra_categoricals']['mappings'][self.class_label]

        if level == 'cell':
            y_true = adata.obsm['_scvi_extra_categoricals'][self.class_label].values
            y_pred = adata.obs['predicted_class'].values    
            print(classification_report(y_true, y_pred, target_names=target_names))
        elif level == 'bag':
            y_true = adata.uns['bag_true']
            y_pred = adata.uns['bag_pred']
            print(classification_report(y_true, y_pred, target_names=target_names))
        elif level == 'patient':
            y_true = pd.DataFrame(adata.obs[self.patient_column]).join(pd.DataFrame(adata.obsm['_scvi_extra_categoricals'][self.class_label])).groupby(self.patient_column).agg('first')
            y_true = y_true[self.class_label].values
            y_pred = adata.obs[[self.patient_column, 'predicted_class']].groupby(self.patient_column).agg(lambda x:x.value_counts().index[0])
            y_pred = y_pred['predicted_class'].values
            print(classification_report(y_true, y_pred, target_names=target_names))
        else:
            raise RuntimeError(f"level={level} not in ['patient', 'bag', 'cell'].")

    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: BaseModelClass,
        use_gpu: Optional[Union[str, int, bool]] = None,
        freeze: bool = True
    ):
        attr_dict = reference_model._get_user_attributes()
        attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}

        model = _initialize_model(cls, adata, attr_dict)

        scvi_setup_dict = attr_dict.pop("scvi_setup_dict_")
        transfer_anndata_setup(scvi_setup_dict, adata, extend_categories=True)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

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
            n_layers_encoders = reference_model.module.vae.n_layers_encoders,
            n_layers_decoders = reference_model.module.vae.n_layers_decoders,
            n_layers_shared_decoder = reference_model.module.vae.n_layers_shared_decoder,
            n_hidden_encoders = reference_model.module.vae.n_hidden_encoders,
            n_hidden_decoders = reference_model.module.vae.n_hidden_decoders,
            n_hidden_shared_decoder = reference_model.module.vae.n_hidden_shared_decoder,
            add_shared_decoder = reference_model.module.vae.add_shared_decoder,
            cont_cov_type = reference_model.module.vae.cont_cov_type,
            n_hidden_cont_embed = reference_model.module.vae.n_hidden_cont_embed,
            n_layers_cont_embed = reference_model.module.vae.n_layers_cont_embed,
            ignore_categories = reference_model.class_label
        )

        vae.module.load_state_dict(reference_model.module.vae.state_dict())

        new_vae = MultiVAE.load_query_data(
            adata,
            reference_model=vae,
            use_gpu=use_gpu,
            freeze=freeze,
        )

        model.module = reference_model.module
        model.module.vae = new_vae.module

        use_gpu, device = parse_use_gpu_arg(use_gpu)
        model.to_device(device)

        if freeze:
            for name, p in model.module.named_parameters():
                if 'vae' not in name:
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
    ):
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
            n_layers_encoders = self.module.vae.n_layers_encoders,
            n_layers_decoders = self.module.vae.n_layers_decoders,
            n_layers_shared_decoder = self.module.vae.n_layers_shared_decoder,
            n_hidden_encoders = self.module.vae.n_hidden_encoders,
            n_hidden_decoders = self.module.vae.n_hidden_decoders,
            n_hidden_shared_decoder = self.module.vae.n_hidden_shared_decoder,
            add_shared_decoder = self.module.vae.add_shared_decoder,
            cont_cov_type = self.module.vae.cont_cov_type,
            n_hidden_cont_embed = self.module.vae.n_hidden_cont_embed,
            n_layers_cont_embed = self.module.vae.n_layers_cont_embed,
            ignore_categories = self.class_label
        )

        vae.module.load_state_dict(self.module.vae.state_dict())

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
            vae.plot_losses()

        self.module.vae = vae.module
        self.is_trained_ = True