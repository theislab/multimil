import numpy as np
from scipy import sparse
import torch
from .scdataset import SingleCellDataset

class SingleCellDatasetMIL(SingleCellDataset):

    def __init__(
        self,
        adata,
        name,
        modality,
        pair_group,
        celltype_key='cell_type',
        batch_size=64,
        batch_label=None,
        layer=None,
        label=None
    ):

        super().__init__(adata,
        name,
        modality,
        pair_group,
        celltype_key,
        batch_size,
        batch_label,
        layer)

        self.label = label

    def _collate_fn(self, batch):
        x = [b[0] for b in batch]
        celltype = [b[1] for b in batch]
        indices = [b[2][0] for b in batch]
        size_factors = [b[3] for b in batch]
        x = torch.stack(x, dim=0)
        size_factors = torch.Tensor(size_factors)
        return x, self.name, self.modality, self.pair_group, celltype, indices, self.label, size_factors, self.batch_label
