from math import ceil, floor
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, Dataset

from scvi import _CONSTANTS, settings
from ..dataloaders._ann_dataloader import BagAnnDataLoader
from scvi.model._utils import parse_use_gpu_arg
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.dataloaders import DataSplitter

class BagDataSplitter(DataSplitter):
    def __init__(
        self,
        adata: AnnData,
        class_column: str,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.class_column = class_column
        super().__init__(adata,
                        train_size,
                        validation_size,
                        use_gpu,
                        **kwargs)

    def train_dataloader(self):
        return BagAnnDataLoader(
            self.adata,
            self.class_column,
            indices=self.train_idx,
            shuffle=True,
            drop_last=3,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return BagAnnDataLoader(
                self.adata,
                self.class_column,
                indices=self.val_idx,
                shuffle=True,
                drop_last=3,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return BagAnnDataLoader(
                self.adata,
                self.class_column,
                indices=self.test_idx,
                shuffle=True,
                drop_last=3,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
