import copy
import itertools

import numpy as np
import torch
from scvi.data import AnnDataManager
from scvi.dataloaders import AnnTorchDataset
from torch.utils.data import DataLoader


# adjusted from scvi-tools
# https://github.com/YosefLab/scvi-tools/blob/ac0c3e04fcc2772fdcf7de4de819db3af9465b6b/scvi/dataloaders/_ann_dataloader.py#L15
# accessed on 4 November 2021
class StratifiedSampler(torch.utils.data.sampler.Sampler):
    """Custom stratified sampler class which enables sampling the same number of observation from each group in each mini-batch.

    Parameters
    ----------
    indices
        List of indices to sample from.
    batch_size
        Batch size of each iteration.
    shuffle
        If ``True``, shuffles indices before sampling.
    drop_last
        If int, drops the last batch if its length is less than drop_last.
        If drop_last == True, drops last non-full batch.
        If drop_last == False, iterate over all batches.
    """

    def __init__(
        self,
        indices: np.ndarray,
        group_labels: np.ndarray,
        batch_size: int,
        min_size_per_class: int,
        shuffle: bool = True,
        drop_last: bool | int = True,
        shuffle_classes: bool = True,
    ):
        if drop_last > batch_size:
            raise ValueError(
                "drop_last can't be greater than batch_size. "
                + f"drop_last is {drop_last} but batch_size is {batch_size}."
            )

        if batch_size % min_size_per_class != 0:
            raise ValueError(
                "min_size_per_class has to be a divisor of batch_size."
                + f"min_size_per_class is {min_size_per_class} but batch_size is {batch_size}."
            )

        self.indices = indices
        self.group_labels = group_labels
        self.n_obs = len(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_classes = shuffle_classes
        self.min_size_per_class = min_size_per_class
        self.drop_last = drop_last

        from math import ceil

        classes = list(dict.fromkeys(self.group_labels))

        tmp = 0
        for cl in classes:
            idx = np.where(self.group_labels == cl)[0]
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

        classes = list(dict.fromkeys(self.group_labels))
        data_iter = []

        for cl in classes:
            idx = np.where(self.group_labels == cl)[0]
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
                idx = idx[:-drop_last_n]

            data_iter.extend(
                [cl_idx[idx[i : i + self.min_size_per_class]] for i in range(0, len(idx), self.min_size_per_class)]
            )

        if self.shuffle_classes:
            idx = torch.randperm(len(data_iter)).tolist()
            data_iter = [data_iter[id] for id in idx]

        final_data_iter = []

        end = len(data_iter) - len(data_iter) % classes_per_batch
        for i in range(0, end, classes_per_batch):
            batch_idx = list(itertools.chain.from_iterable(data_iter[i : i + classes_per_batch]))
            final_data_iter.append(batch_idx)

        # deal with the last manually
        if end != len(data_iter):
            batch_idx = list(itertools.chain.from_iterable(data_iter[end:]))
            final_data_iter.append(batch_idx)

        return iter(final_data_iter)

    def __len__(self):
        return self.length


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/dataloaders/_ann_dataloader.py#L89
# accessed on 5 November 2022
class GroupAnnDataLoader(DataLoader):
    """DataLoader for loading tensors from AnnData objects.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object with a registered AnnData object.
    shuffle
        Whether the data should be shuffled.
    indices
        The indices of the observations in the adata to load.
    batch_size
        Minibatch size to load each iteration.
    data_and_attributes
        Dictionary with keys representing keys in data registry (`adata.uns["_scvi"]`)
        and value equal to desired numpy loading type (later made into torch tensor).
        If `None`, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        group_column: str,
        shuffle=True,
        shuffle_classes=True,
        indices=None,
        batch_size=128,
        min_size_per_class=None,
        data_and_attributes: dict | None = None,
        drop_last: bool | int = True,
        sampler: torch.utils.data.sampler.Sampler | None = StratifiedSampler,
        **data_loader_kwargs,
    ):
        if adata_manager.adata is None:
            raise ValueError("Please run register_fields() on your AnnDataManager object first.")

        if data_and_attributes is not None:
            data_registry = adata_manager.data_registry
            for key in data_and_attributes.keys():
                if key not in data_registry.keys():
                    raise ValueError(f"{key} required for model but not registered with AnnDataManager.")

        if group_column not in adata_manager.registry["setup_args"]["categorical_covariate_keys"]:
            raise ValueError(
                f"{group_column} required for model but not in categorical covariates has to be one of the registered categorical covariates = {adata_manager.registry['setup_args']['categorical_covariate_keys']}."
            )

        self.dataset = AnnTorchDataset(adata_manager, getitem_tensors=data_and_attributes)

        if min_size_per_class is None:
            min_size_per_class = batch_size // 2

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "min_size_per_class": min_size_per_class,
            "shuffle_classes": shuffle_classes,
        }

        if indices is None:
            indices = np.arange(len(self.dataset))
            sampler_kwargs["indices"] = indices
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
            sampler_kwargs["indices"] = indices

        sampler_kwargs["group_labels"] = np.array(
            adata_manager.adata[indices].obsm["_scvi_extra_categorical_covs"][group_column]
        )

        self.indices = indices
        self.sampler_kwargs = sampler_kwargs

        sampler = sampler(**self.sampler_kwargs)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        super().__init__(self.dataset, **self.data_loader_kwargs)
