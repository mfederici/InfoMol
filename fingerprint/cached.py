import os
from typing import List

import numpy as np

from caching.cached_dict import CachedTensorDict
from fingerprint.base import Fingerprinter


class CachedFingerprinter(Fingerprinter):
    def __init__(self, cache_path: str, fingerprinter: Fingerprinter, write_size: int = 1000):
        self.name = fingerprinter.name
        self.n_dim = fingerprinter.n_dim
        super().__init__(verbose=fingerprinter.verbose)
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache_path = cache_path
        self.cache = CachedTensorDict(cache_path, self.name, shape=(self.n_dim,), write_size=write_size)
        self.fingerprinter = fingerprinter

    def encode(self, smile: str) -> np.ndarray:
        # Read from cache
        if smile in self.cache:
            value = self.cache[smile]
        # Compute if necessary
        else:
            value = self.fingerprinter.encode(smile)
            self.cache[smile] = value

        return value

    def encode_all(self, smiles: List[str]) -> np.ndarray:
        # Determine which smile still have to be computed
        smiles_to_compute = []
        for smile in smiles:
            if not (smile in self.cache):
                smiles_to_compute.append(smile)

        # Compute them all at once
        if len(smiles_to_compute) > 0:
            new_fingerprints = self.fingerprinter.encode_all(smiles_to_compute)
            assert new_fingerprints.shape[0] == len(smiles_to_compute)
            # Update the cache with the new values
            for i, smile in enumerate(smiles_to_compute):
                self.cache[smile] = new_fingerprints[i]

        # Read up all values from the cache
        return self.cache[smiles]
    
    def write_cache_to_disk(self):
        self.cache.write_cache_to_disk()

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += f"  (fingerprinter): "+self.fingerprinter.__repr__().replace('\n','\n  ')+"\n"
        s += f"  (cache_path): "+self.cache.path + "\n)"
        return s
