import os
from typing import Dict, Any, List, Optional, Callable, Type

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.data import BaseData


class FishTox(InMemoryDataset):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 ):
        filepath = os.path.join(root, 'fishtox.csv')
        if not os.path.isfile(filepath):
            raise ValueError("The root does not contain the fishtox.csv file")
        super().__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = self.load(filepath)


    def load(self, path: str, data_cls: Type[BaseData] = Data):
        dataframe = pd.read_csv(path)
        self.y = torch.FloatTensor(dataframe['LC50'].values).reshape(-1,1)
        self.smiles = list(dataframe['SMILES'].values)
        slices = {'smiles': self.smiles, 'y': self.y}
        return data_cls(**slices), slices

    def __len__(self):
        return len(self.smiles)

