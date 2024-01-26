import os.path
from typing import Tuple, Optional, List, Union
import os
import h5py
import pickle
import tables

import numpy as np


class CachedTensorDict:
    DATASET_NAME = 'data'
    TENSOR_FILE = 'values.h5py'
    INDEX_FILE = 'index.pkl'
    def __init__(self, path: str, name:str, shape: Tuple[int, ...], write_size: int = 1000):
        assert os.path.isdir(path)

        full_path = os.path.join(path, name)
        if not os.path.isdir(full_path):
            os.mkdir(full_path)

        self.path = full_path
        self.tensor_file = os.path.join(full_path, self.TENSOR_FILE)
        self.index_file = os.path.join(full_path, self.INDEX_FILE)

        self.write_size = write_size
        self._base_shape = shape

        if not os.path.isfile(self.index_file):
            with h5py.File(self.tensor_file, 'w') as f:
                f.create_dataset(
                    self.DATASET_NAME,
                    shape=(0, *shape),
                    maxshape=(None, *shape)
                )
            with open(self.index_file, 'wb') as file:
                pickle.dump({}, file)

        self.f = h5py.File(self.tensor_file, 'r')
        self._cache = self.f[self.DATASET_NAME]

        with open(self.index_file, 'rb') as file:
            self._index = pickle.load(file)
        self._new_cache = {}

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self._index), *self._base_shape)

    def write_cache_to_disk(self):
        if len(self._new_cache) == 0:
            return

        # Write the freshly added data
        old_cache_size = self._cache.shape[0]
        new_cache_size = len(self._index)

        # Change open mode
        self.f.close()
        self.f = h5py.File(self.tensor_file, 'w')
        self._cache = self.f[self.DATASET_NAME]

        # Resize to fit the new data
        self._cache.resize((new_cache_size, *self._base_shape))

        # Write the new values
        for smile, value in self._new_cache.items():
            # Empty entry
            if not value is None:
                idx = self._index[smile]
                assert idx >= old_cache_size, f"{idx}, {old_cache_size}"
                self._cache[idx] = value

        # Change back to read mode
        self.f.close()
        self.f = h5py.File(self.tensor_file, 'r')
        self._cache = self.f[self.DATASET_NAME]

        # Update the index
        with open(self.index_file, 'wb') as f:
            pickle.dump(self._index, f)

        self._new_cache = {}

    def _add_to_cache(self, key: str, value: Optional[np.ndarray]):
        assert not (key in self._index) and not (key in self._new_cache)

        if value is None:
            self._index[key] = -1
        else:
            self._index[key] = len(self._index)
        self._new_cache[key] = value

        # Write to disk when the data in memory is too much
        if len(self._index)-len(self._cache) >= self.write_size:
            self.write_cache_to_disk()


    def __contains__(self, key: str):
        return key in self._index or key in self._new_cache

    def __getitem__(self, key: Union[str, List[str]]) -> np.ndarray:
        if isinstance(key, str):
            if key in self._new_cache:
                value = self._new_cache[key]
            else:
                idx = self._index[key]
                value = self._cache[idx]
            return value
        elif isinstance(key, list):
            # Write everithing to cache
            self.write_cache_to_disk()

            # Then look up
            ids = np.array([self._index[k] for k in key])
            if len(ids) > 1:
                # Make sure the ids are ordered
                if not np.all(ids[:-1] <= ids[1:]):
                    # Step 1: Sort the IDs and remember the original indices
                    id_order = np.argsort(ids)
                    sorted_ids = ids[id_order]

                    # Step 2: Access the array using the sorted IDs
                    values = self._cache[sorted_ids]

                    # Step 3: Sort the values back to the original order
                    inverse_indices = np.empty_like(ids)
                    inverse_indices[id_order] = np.arange(len(key))
                    values = values[inverse_indices]
                else:
                    values = self._cache[ids]
            else:
                values = self._cache[ids]
            return values


    def __len__(self) -> int:
        return len(self._index)

    def __setitem__(self, key: str, value: Optional[np.ndarray]):
        assert value is None or value.shape == self._base_shape

        if not key in self:
            self._add_to_cache(key, value)
        elif key in self._new_cache:
            self._new_cache[key] = value
        else:
            idx = self._index[key]
            self._cache[idx] = value

    def __del__(self):
        self.write_cache_to_disk()
        self.f.close()

    def __bool__(self):
        return True