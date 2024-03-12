import os
from typing import List

import numpy as np
from tqdm.auto import tqdm

from caching.cached_dict import CachedTensorDict
from encoder.base import Encoder
from utils.batching import make_batches


class CachedEncoder(Encoder):

    @staticmethod
    def get_name(encoder: Encoder, **kwargs) -> str:
        return encoder.name

    @staticmethod
    def get_n_dim(encoder: Encoder, **kwargs) -> int:
        return encoder.n_dim

    def __init__(self, cache_path: str, encoder: Encoder, write_size: int = 1000, read_only: bool = False):
        self.n_dim = encoder.n_dim
        super().__init__(verbose=encoder.verbose, encoder=encoder)
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache_path = cache_path
        self.cache = CachedTensorDict(cache_path, self.name, shape=(self.n_dim,), write_size=write_size, read_only=read_only)
        self.encoder = encoder

    def encode(self, smile: str) -> np.ndarray:
        # Read from cache
        if smile in self.cache:
            value = self.cache[smile]
        # Compute if necessary
        else:
            value = self.encoder.encode(smile)
            self.cache[smile] = value

        return value

    def encode_all(self, smiles: List[str]) -> np.ndarray:
        # Determine which smile still have to be computed
        smiles_to_compute = []
        for smile in smiles:
            if not (smile in self.cache):
                smiles_to_compute.append(smile)

        # Compute in batches of self.cache.write_size
        if len(smiles_to_compute) > 0:
            smiles_batches = make_batches(smiles_to_compute, self.cache.write_size)

            if self.verbose:
                print(f"Computing {len(smiles_to_compute)} new fingerprints.")
                smiles_batches = tqdm(smiles_batches)

            for smiles_batch in smiles_batches:
                new_fingerprints = self.encoder.encode_all(smiles_batch)
                assert new_fingerprints.shape[0] == len(smiles_batch), f"{new_fingerprints.shape[0]}!={len(smiles_batch)}"
                # Update the cache with the new values
                for i, smile in enumerate(smiles_batch):
                    self.cache[smile] = new_fingerprints[i]

        fingerprints = self.cache[smiles]

        # Read up all values from the cache
        return fingerprints
    
    def write_cache_to_disk(self):
        self.cache.write_cache_to_disk()

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += f"  (encoder): " + self.encoder.__repr__().replace('\n', '\n  ') + "\n"
        s += f"  (cache_path): "+self.cache.path + "\n)"
        return s
