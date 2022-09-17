from typing import Optional
from anndata import AnnData

from ..dataloaders._ann_dataloader import GroupAnnDataLoader
from scvi.dataloaders import DataSplitter


class GroupDataSplitter(DataSplitter):
    def __init__(
        self,
        adata: AnnData,
        group_column: str,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.group_column = group_column
        super().__init__(adata, train_size, validation_size, use_gpu, **kwargs)

    def train_dataloader(self):
        return GroupAnnDataLoader(
            self.adata,
            self.group_column,
            indices=self.train_idx,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return GroupAnnDataLoader(
                self.adata,
                self.group_column,
                indices=self.val_idx,
                shuffle=True,
                drop_last=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return GroupAnnDataLoader(
                self.adata,
                self.group_column,
                indices=self.test_idx,
                shuffle=True,
                drop_last=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
