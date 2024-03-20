from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np
from tqdm.auto import tqdm
import os

from caching.cached_dict import CachedTensorDict
from utils.batching import make_batches

WRITE_SIZE = 1000
DATASET_NAME = 'fingerprints'

DEEP_ENCODER='deep'
FINGERPRINT_ENCODER='fingerprint'
PROPERTY_ENCODER='property'

class Encoder:
    n_dim: Optional[int] = None
    name: Optional[str] = None
    type: Optional[str] = None

    def _nan_vector(self) -> np.ndarray:
        empty_vector = np.empty([self.n_dim])
        empty_vector[:] = np.nan
        return empty_vector

    @staticmethod
    def get_name(**kwargs) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        raise NotImplemented()

    def __init__(self, verbose: bool = False, cache_path: Optional[str] = None, write_size=1000, read_only: bool=False, **kwargs):
        self.verbose = verbose
        self.name = self.get_name(**kwargs)
        self.n_dim = self.get_n_dim(**kwargs)

        if cache_path is None:
            self.cache = None
        else:
            if not os.path.isdir(cache_path):
                os.mkdir(cache_path)
            self.cache = CachedTensorDict(cache_path, self.name, shape=(self.n_dim,), write_size=write_size,
                                          read_only=read_only)


    @abstractmethod
    def _encode_one(self, smile:str) -> np.ndarray:
        raise NotImplementedError()

    def encode_one(self, smile: str) -> np.ndarray:
        if self.cache is None:
            value = self._encode_one(smile)
        # Read from cache
        elif smile in self.cache:
            value = self.cache[smile]
        # Compute if necessary
        else:
            value = self._encode_one(smile)

            # Add the value to the cache if using caching
            self.cache[smile] = value

        return value

    def _encode_all(self, smiles: List[str]) -> np.ndarray:
        representations = []
        if self.verbose:
            smiles = tqdm(smiles)

        for smile in smiles:
            representations.append(self.encode_one(smile).reshape(1, -1))

        representations = np.concatenate(representations, 0)
        assert representations.shape[0] == len(smiles)

        return representations

    def encode_all(self, smiles: List[str]) -> np.ndarray:
        if self.cache is None:
            representations = self._encode_all(smiles)
        else:
            # Determine which smile still have to be computed
            smiles_to_compute = []
            for smile in smiles:
                if not (smile in self.cache):
                    smiles_to_compute.append(smile)

            # Compute in batches of self.cache.write_size
            if len(smiles_to_compute) > 0:
                smiles_batches = make_batches(smiles_to_compute, self.cache.write_size)

                if self.verbose:
                    print(f"Computing {len(smiles_to_compute)} new representations.")
                    smiles_batches = tqdm(smiles_batches)

                for smiles_batch in smiles_batches:
                    new_fingerprints = self._encode_all(smiles_batch)
                    assert new_fingerprints.shape[0] == len(
                        smiles_batch), f"{new_fingerprints.shape[0]}!={len(smiles_batch)}"
                    # Update the cache with the new values
                    for i, smile in enumerate(smiles_batch):
                        self.cache[smile] = new_fingerprints[i]

            # Read up all values from the cache
            representations = self.cache[smiles]

        return representations

    def write_cache_to_disk(self):
        if self.cache:
            self.cache.write_cache_to_disk()

    def __call__(self, smiles: Union[str, List[str]]) -> np.ndarray:
        if isinstance(smiles, str):
            return self.encode_one(smiles)
        else:
            return self.encode_all(smiles)
