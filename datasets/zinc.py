from typing import Any, Callable, Optional

import numpy as np
import torch
from torch_geometric.data import download_url
import os
from torch.utils.data import Dataset


class ZINCSmiles(Dataset):
    URL = "https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/all.txt"
    SMILES_FILE = 'all.txt'

    def __init__(self, root: str = '/data'):

        data_dir = os.path.join(root, 'ZINC')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        smiles_filepath = os.path.join(data_dir, self.SMILES_FILE)
        if not os.path.isfile(smiles_filepath):
            download_url(self.URL, data_dir)

        with open(smiles_filepath, 'r') as f:
            smiles = f.readlines()
        self.smiles = [s.replace('\n', '') for s in smiles]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        return self.smiles[item]

    def __repr__(self):
        return f"ZINCSmiles({self.__len__()})"
