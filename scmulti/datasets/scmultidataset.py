import torch
import anndata
import numpy as np
from collections import OrderedDict
from scipy import sparse
import os


class SingleCellMultiDataset(torch.utils.data.Dataset):

    def __init__(self, adatas, xs, pair_indices):
        self.adatas = adatas
        self.xs = xs
        self.pair_indices = pair_indices
        self.names = list(self.adatas.keys())
    
    def __len__(self):
        return max([len(x) for x in self.xs.values()])

    def __getitem__(self, idx):
        xs = [x[idx % len(x)] for x in self.xs.values()]
        pair_indices = [pair_idx[idx % len(pair_idx)] for pair_idx in self.pair_indices.values()]
        return xs, pair_indices


class SingleCellMultiDatasetBuilder:

    def __init__(self, val_split=.2, pair_split=0, device='cpu', random_seed=42):
        self.pair_split = pair_split
        self.val_split = val_split
        self.random_seed = random_seed
        self.device = device

        self.adatas = OrderedDict()
        self.xs = OrderedDict()
        self.pair_groups = OrderedDict()  # maps each paired dataset's name to a pair group id; `None` if unpaired
        self.pair_indices = OrderedDict()  # maps each dataset's name to a one hot vector which indicates paired samples
        self.val_indices = OrderedDict()  # maps each dataset's name to a one hot vector which indicates validation samples
        self.group_pair_indices = {}  # maps each pair group id to a one hot vector which indicates paired samples
        self.group_val_indices = {}  # maps each pair group id to a one hot vector which indicates validation samples


    def add_dataset(self, name, h5ad_path, pair_group=None):
        adata = self.load_h5ad(h5ad_path)
        dataset_size = len(adata)

        self.adatas[name] = adata
        self.xs[name] = self._create_tensor(adata.X).to(self.device)
        self.pair_groups[name] = pair_group

        # set validation indices
        self._set_random_val_indices(name, dataset_size=dataset_size, val_size=int(dataset_size * self.val_split))

        # set pair indices
        if pair_group is not None:
            # the dataset is paired
            self._set_random_pair_indices(name, dataset_size=dataset_size, pair_size=int(dataset_size * self.pair_split))
        else:
            # the dataset is unpaired, just set all indices to zero
            self._set_random_pair_indices(name, dataset_size=dataset_size, pair_size=0)
    

    def load_h5ad(self, path):
        return anndata.read_h5ad(path)
    

    def _set_random_val_indices(self, name, dataset_size, val_size):
        pair_group = self.pair_groups[name]
        # check if the dataset is paired and the validation indices are already set for its pair group
        if pair_group is not None and pair_group in self.group_val_indices: 
            self.val_indices[name] = self.group_val_indices[pair_group]
        else:
            # the dataset is either unpaired or paired with no previous validation indices set for its pair group
            # so we generate new random indices
            random_indices = np.zeros(dataset_size, dtype=np.uint8)
            random_indices[:val_size] = 1
            np.random.shuffle(random_indices)
            self.val_indices[name] = random_indices
            if pair_group is not None:
                # the dataset is paired
                # memoize the indices for the group
                self.group_val_indices[pair_group] = random_indices


    def _set_random_pair_indices(self, name, dataset_size, pair_size):
        pair_group = self.pair_groups[name]
        # check if the dataset is paired and the paired indices are already set for its pair group
        if pair_group is not None and pair_group in self.group_pair_indices: 
            self.pair_indices[name] = self.group_pair_indices[pair_group]
        else:
            # the dataset is either unpaired or paired with no previous pair indices set for its pair group
            # so we generate new random indices
            random_indices = np.zeros(dataset_size, dtype=np.uint8)
            random_indices[:pair_size] = 1
            np.random.shuffle(random_indices)
            self.pair_indices[name] = random_indices
            if pair_group is not None:
                # the dataset is paired
                # memoize the indices for the group
                self.group_pair_indices[pair_group] = random_indices


    def _create_tensor(self, x):
        if sparse.issparse(x):
            x = x.todense()
            return torch.FloatTensor(x)
        else:
            return torch.FloatTensor(x)
    

    def build(self):
        train_adatas, test_adatas = OrderedDict(), OrderedDict()
        train_xs, test_xs = OrderedDict(), OrderedDict()
        train_pair_indices, test_pair_indices = OrderedDict(), OrderedDict()

        for name in self.adatas:
            adata, x = self.adatas[name], self.xs[name]
            pair_indices, val_indices = self.pair_indices[name], self.val_indices[name]
            
            train_adatas[name], test_adatas[name] = adata[val_indices == 0], adata[val_indices == 1]
            # BUG: torch.Tensor does not accept np.ndarray mask in the pytorch version I'm using
            # so I had to convert the np.ndarray mask to a torch.Tensor mask
            train_xs[name], test_xs[name] = x[torch.from_numpy(val_indices) == 0], x[torch.from_numpy(val_indices) == 1]
            train_pair_indices[name], test_pair_indices[name] = pair_indices[val_indices == 0], pair_indices[val_indices == 1]
        
        train_dataset = SingleCellMultiDataset(train_adatas, train_xs, train_pair_indices)
        test_dataset = SingleCellMultiDataset(test_adatas, test_xs, test_pair_indices)
        return train_dataset, test_dataset