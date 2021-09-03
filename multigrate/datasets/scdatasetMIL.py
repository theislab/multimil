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
        self.adata = adata
        self.name = name
        self.modality = modality
        self.pair_group = pair_group
        self.celltype_key = celltype_key
        self.batch_label = batch_label
        self.label = label

        if layer:
            self.x = self._create_tensor(adata.layers[layer])
            self.size_factors = adata.layers[layer].sum(1)
        else:
            self.x = self._create_tensor(adata.X)
            self.size_factors = adata.X.sum(1)

        #self.size_factors = np.log(adata.X.sum(1))

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
        indices = [b[2][0] for b in batch]
        size_factors = [b[3] for b in batch]
        x = torch.stack(x, dim=0)
        size_factors = torch.Tensor(size_factors)
        return x, self.name, self.modality, self.pair_group, celltype, indices, self.label, size_factors, self.batch_label

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        return self.x[idx], self.adata[idx].obs[self.celltype_key].item(), self.adata[idx].obs_names, self.size_factors[idx].item()
