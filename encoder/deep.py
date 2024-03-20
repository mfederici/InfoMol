from abc import abstractmethod
from typing import List, Union, Optional

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data

from encoder import Encoder
from encoder.base import DEEP_ENCODER


class DeepEncoder(Encoder):
    POOL : List[str] = []
    DEFAULT_POOL = 'mean'
    type: str = DEEP_ENCODER

    def __init__(self, pool: Optional[str] = DEFAULT_POOL, device: Union[torch.device, str] = 'cpu', **kwargs):
        if len(self.POOL) >0 and not (pool in self.POOL):
            raise ValueError(f'The available pool stratefies are {self.POOL} (default: "mean")')
        self.pool = pool

        super().__init__(pool=pool, **kwargs)
        if self.verbose:
            print(f'Instantiating the model.')
        model = self._instantiate_model()

        self.device = device
        self.model = model.to(device).eval()

        if self.verbose:
            print("Model loaded")

    @abstractmethod
    def _instantiate_model(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def _model_encode_smile(self, smile: str) -> torch.Tensor:
        raise NotImplementedError()

    def _encode_one(self, smile: str) -> np.ndarray:
        try:
            representation = self._model_encode_smile(smile)
            representation = representation.to('cpu').data.numpy()
        except Exception as e:
            print(f'Error while computing the representation of {smile}')
            print(e)
            return self._nan_vector()

        assert representation.shape[-1] == self.n_dim
        assert representation.ndim == 1
        return representation


class DeepGraphEncoder(DeepEncoder):
    @abstractmethod
    def _mol2graph(self, mol: Chem.Mol) -> Data:
        raise NotImplementedError()

    @abstractmethod
    def _model_encode_graph(self, graph: Data)-> torch.Tensor:
        raise NotImplementedError()

    def _model_encode_smile(self, smile: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid Molecule {smile}")

        graph = self._mol2graph(mol).to(self.device)
        with torch.no_grad():
            representation = self._model_encode_graph(graph)

        return representation
