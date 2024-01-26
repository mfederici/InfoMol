from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

WRITE_SIZE = 1000
DATASET_NAME = 'fingerprints'

class Fingerprinter:
    n_dim: Optional[int] = None
    name: Optional[str] = None

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        assert self.n_dim
        assert self.name

    def encode_all(self, smiles: List[str]) -> np.ndarray:
        fps = []
        if self.verbose:
            smiles = tqdm(smiles)

        for smile in smiles:
            fps.append(self.encode(smile).reshape(1,-1))

        fps = np.concatenate(fps, 0)
        assert fps.shape[0] == len(smiles)

        return fps

    @abstractmethod
    def encode(self, smile: str) -> np.ndarray:
        raise NotImplemented()

    def __call__(self, smiles: Union[str, List[str]]) -> np.ndarray:
        if isinstance(smiles, str):
            return self.encode(smiles)
        else:
            return self.encode_all(smiles)
