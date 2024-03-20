import os
from typing import Dict, Any, List, Optional, Callable, Type, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.data.data import BaseData

# Original dataset from https://uvaauas.figshare.com/ndownloader/files/35936597
class FishTox(Dataset):
    URL = "https://raw.githubusercontent.com/mfederici/InfoMol/master/data/toxicity_data_fish_desc.csv"
    FILENAME = "toxicity_data_fish_desc.csv"
    def __init__(self,
                 root: str,
                 ):
        data_dir = os.path.join(root, 'FishTox')
        data_filepath = os.path.join(data_dir, self.FILENAME)

        if not os.path.isfile(data_filepath):
            download_url(self.URL, data_dir)

        dataframe = pd.read_csv(data_filepath)
        self.y = dataframe['LC50[-LOG(mol/L)]'].values.reshape(-1, 1).astype(np.float32)
        self.smiles = list(dataframe['SMILES'].values)

    def __getitem__(self, item) -> Tuple[str, float]:
        return self.smiles[item], self.y[item]

    def __len__(self):
        return len(self.smiles)

