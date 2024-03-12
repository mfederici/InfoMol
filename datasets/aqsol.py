import numpy as np
import pandas as pd
from torch_geometric.data import download_url
import os
from torch.utils.data import Dataset


class AQSOLSmiles(Dataset):
    URL = "https://raw.githubusercontent.com/Mengjintao/SolCuration/master/org/aqsol/aqsol_org.csv"
    DATA_FILE = 'aqsol_org.csv'

    def __init__(self, root: str = '/data'):

        data_dir = os.path.join(root, 'AQSOL')
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        data_filepath = os.path.join(data_dir, self.DATA_FILE)
        if not os.path.isfile(data_filepath):
            download_url(self.URL, data_dir)

        data = pd.read_csv(data_filepath)
        self.smiles = data['smiles'].values
        self.y = data['logS'].values.reshape(-1,1).astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        if hasattr(self, 'y'):
            return self.smiles[item], self.y[item]
        else:
            return self.smiles[item]

    def __repr__(self):
        return f"AQSOLSmiles({self.__len__()})"
