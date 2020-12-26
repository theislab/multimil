import numpy as np
from scipy import sparse
import torch


class SingleCellDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        adata,
        name,
        modality,
        pair_group,
        celltype_key='cell_type',
        batch_size=64,
        batch_label=None,
        layer=None
    ):
        self.adata = adata
        self.name = name
        self.modality = modality
        self.pair_group = pair_group
        self.celltype_key = celltype_key
        self.batch_label = batch_label

        if layer:
            self.x = self._create_tensor(adata.layers[layer])
        else:
            self.x = self._create_tensor(adata.X)

        self.loader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

    def _create_tensor(self, x):
        if sparse.issparse(x):
            x = x.todense()
            return torch.FloatTensor(x)
        else:
            return torch.FloatTensor(x)

    def _collate_fn(self, batch):
        x = [b[0] for b in batch]
        celltype = [b[1] for b in batch]
        x = torch.stack(x, dim=0)
        return x, self.name, self.modality, self.pair_group, celltype, self.batch_label

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        return self.x[idx], self.adata[idx].obs[self.celltype_key].item()
