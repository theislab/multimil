import numpy as np
from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter

from multimil.dataloaders._ann_dataloader import GroupAnnDataLoader


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/dataloaders/_data_splitting.py#L56
# accessed on 5 November 2022
class GroupDataSplitter(DataSplitter):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    Splits are computed at the **sample level** (i.e. every cell that belongs to a given
    sample lands entirely in train, validation, or test) rather than at the cell level.
    This guarantees that each split contains complete bags, which is required for the
    grouped dataloader.

    If ``train_size + validation_size < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    group_column
        Key in ``adata.obs`` that identifies samples / bags.
    train_size
        Proportion of **samples** to use as the train set (default 0.9).
    validation_size
        Proportion of **samples** for validation. If ``None``, defaults to
        ``1 - train_size``.
    kwargs
        Keyword args forwarded to :class:`~multimil.dataloaders.GroupAnnDataLoader`.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        group_column: str,
        train_size: float = 0.9,
        validation_size: float | None = None,
        **kwargs,
    ):
        self.group_column = group_column

        # --- sample-level split ------------------------------------------------
        sample_labels = np.asarray(
            adata_manager.adata.obsm["_scvi_extra_categorical_covs"][group_column]
        )
        unique_samples = np.unique(sample_labels)
        n_samples = len(unique_samples)

        # Compute n_train first to avoid floating-point error in 1.0 - train_size
        # (e.g. 1.0 - 0.9 = 0.09999... causing floor to under-count by 1).
        if validation_size is None:
            n_train = int(np.floor(train_size * n_samples))
            n_val = n_samples - n_train
        else:
            n_val = int(np.floor(validation_size * n_samples))
            n_train = n_samples - n_val

        rng = np.random.RandomState(seed=settings.seed)
        perm = rng.permutation(n_samples)
        val_samples = set(unique_samples[perm[:n_val]])
        train_samples = set(unique_samples[perm[n_val : n_val + n_train]])

        all_idx = np.arange(len(sample_labels))
        train_idx = all_idx[np.array([s in train_samples for s in sample_labels])]
        val_idx = all_idx[np.array([s in val_samples for s in sample_labels])]
        test_idx = np.array([], dtype=np.intp)
        # -----------------------------------------------------------------------

        super().__init__(
            adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            external_indexing=[train_idx, val_idx, test_idx],
            **kwargs,
        )

    def train_dataloader(self):
        """Return data loader for train AnnData."""
        return GroupAnnDataLoader(
            self.adata_manager,
            self.group_column,
            indices=self.train_idx,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Return data loader for validation AnnData."""
        if len(self.val_idx) > 0:
            return GroupAnnDataLoader(
                self.adata_manager,
                self.group_column,
                indices=self.val_idx,
                shuffle=False,
                drop_last=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Return data loader for test AnnData."""
        if len(self.test_idx) > 0:
            return GroupAnnDataLoader(
                self.adata_manager,
                self.group_column,
                indices=self.test_idx,
                shuffle=False,
                drop_last=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
