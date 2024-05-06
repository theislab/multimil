from typing import Optional

from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter

from ..dataloaders._ann_dataloader import GroupAnnDataLoader


# adjusted from scvi-tools
# https://github.com/scverse/scvi-tools/blob/0b802762869c43c9f49e69fe62b1a5a9b5c4dae6/scvi/dataloaders/_data_splitting.py#L56
# accessed on 5 November 2022
class GroupDataSplitter(DataSplitter):
    """
    Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    :param adata_manager:
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    :param train_size:
        float, or None (default is 0.9)
    :param validation_size:
        float, or None (default is None)
    :param use_gpu:
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    :param kwargs:
        Keyword args for data loader. Data loader class is :class:`~mtg.dataloaders.GroupAnnDataLoader`.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        group_column: str,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.group_column = group_column
        super().__init__(adata_manager, train_size, validation_size, use_gpu, **kwargs)

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
