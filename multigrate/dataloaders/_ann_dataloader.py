# adjusted from scvi-tools
# https://github.com/YosefLab/scvi-tools/blob/ac0c3e04fcc2772fdcf7de4de819db3af9465b6b/scvi/dataloaders/_ann_dataloader.py#L88
# accessed on 4 November 2021
import numpy as np
import torch
from torch.utils.data import DataLoader
from scvi.dataloaders import AnnTorchDataset
from typing import List, Optional, Union
import anndata
import copy
import itertools

class StratifiedSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        indices: np.ndarray,
        patient_labels: np.ndarray,
        batch_size: int,
        min_size_per_class: int,
        shuffle: bool = True,
        drop_last: Union[bool, int] = True,
        shuffle_classes: bool = True
    ):
        if drop_last > batch_size:
            raise ValueError(
                "drop_last can't be greater than batch_size. "
                + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
            )

        if batch_size % min_size_per_class != 0:
            raise ValueError(
                "min_size_per_class has to be a divisor of batch_size."
                + "min_size_per_class is {} but batch_size is {}.".format(min_size_per_class, batch_size)
            )

        self.indices = indices
        self.patient_labels = patient_labels
        self.n_obs = len(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_classes = shuffle_classes
        self.min_size_per_class = min_size_per_class 
        self.drop_last = drop_last

        from math import ceil

        classes = set(self.patient_labels)

        tmp = 0
        for cl in classes:
            idx = np.where(self.patient_labels == cl)[0]
            cl_idx = self.indices[idx]
            n_obs = len(cl_idx)

            last_batch_len = n_obs % self.min_size_per_class
            if (self.drop_last is True) or (last_batch_len < self.drop_last):
                drop_last_n = last_batch_len
            elif (self.drop_last is False) or (last_batch_len >= self.drop_last):
                drop_last_n = 0
            else:
                raise ValueError("Invalid input for drop_last param. Must be bool or int.")

            if drop_last_n != 0:
                tmp += n_obs // self.min_size_per_class
            else:
                tmp += ceil(n_obs / self.min_size_per_class)

        classes_per_batch = int(self.batch_size / self.min_size_per_class)
        self.length = ceil(tmp / classes_per_batch)


    def __iter__(self):

        classes_per_batch = int(self.batch_size / self.min_size_per_class)

        classes = set(self.patient_labels)
        data_iter = []

        for cl in classes:
            idx = np.where(self.patient_labels == cl)[0]
            cl_idx = self.indices[idx]
            n_obs = len(cl_idx)

            if self.shuffle is True:
                idx = torch.randperm(n_obs).tolist()
            else:
                idx = torch.arange(n_obs).tolist()

            last_batch_len = n_obs % self.min_size_per_class
            if (self.drop_last is True) or (last_batch_len < self.drop_last):
                drop_last_n = last_batch_len
            elif (self.drop_last is False) or (last_batch_len >= self.drop_last):
                drop_last_n = 0
            else:
                raise ValueError("Invalid input for drop_last param. Must be bool or int.")

            if drop_last_n != 0:
                idx = idx[: -drop_last_n]

            data_iter.extend(
            [
                cl_idx[idx[i : i + self.min_size_per_class]]
                for i in range(0, len(idx), self.min_size_per_class)
            ])

        if self.shuffle_classes:
            idx = torch.randperm(len(data_iter)).tolist()
            data_iter = [data_iter[id] for id in idx]

        final_data_iter = []

        end = len(data_iter) - len(data_iter)%classes_per_batch
        for i in range(0, end, classes_per_batch):
            batch_idx = list(itertools.chain.from_iterable(data_iter[i : i+classes_per_batch]))
            final_data_iter.append(batch_idx)

        # deal with the last manually
        if end != len(data_iter):
            batch_idx = list(itertools.chain.from_iterable(data_iter[end:]))
            final_data_iter.append(batch_idx)

        return iter(final_data_iter)

    def __len__(self):
        return self.length

class BagAnnDataLoader(DataLoader):
    """
    DataLoader for loading tensors from AnnData objects.
    Parameters
    ----------
    adata
        An anndata objects
    shuffle
        Whether the data should be shuffled
    indices
        The indices of the observations in the adata to load
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (`adata.uns["_scvi"]`)
        and value equal to desired numpy loading type (later made into torch tensor).
        If `None`, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        patient_column: str,
        shuffle=True,
        shuffle_classes=True,
        indices=None,
        batch_size=128,
        min_size_per_class=None,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = True,
        sampler: Optional[torch.utils.data.sampler.Sampler] = StratifiedSampler,
        **data_loader_kwargs,
    ):

        if "_scvi" not in adata.uns.keys():
            raise ValueError("Please run setup_anndata() on your anndata object first.")

        if data_and_attributes is not None:
            data_registry = adata.uns["_scvi"]["data_registry"]
            for key in data_and_attributes.keys():
                if key not in data_registry.keys():
                    raise ValueError(
                        "{} required for model but not included when setup_anndata was run".format(
                            key
                        )
                    )

        if patient_column not in adata.uns['_scvi']['extra_categoricals']['keys']:
            raise ValueError(
                "{} required for model but not included into categorical covariates when setup_anndata was run".format(
                    patient_column
                )
            )

        self.dataset = AnnTorchDataset(adata, getitem_tensors=data_and_attributes)

        if not min_size_per_class:
            min_size_per_class = batch_size // 2

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "min_size_per_class": min_size_per_class,
            "shuffle_classes": shuffle_classes
        }

        if indices is None:
            indices = np.arange(len(self.dataset))
            sampler_kwargs["indices"] = indices
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
            sampler_kwargs["indices"] = indices

        sampler_kwargs["patient_labels"] = np.array(adata[indices].obsm['_scvi_extra_categoricals'][patient_column])

        self.indices = indices
        self.sampler_kwargs = sampler_kwargs

        sampler = sampler(**self.sampler_kwargs)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        super().__init__(self.dataset, **self.data_loader_kwargs)
