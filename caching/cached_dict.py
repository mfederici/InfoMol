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

    def __init__(self, path: str, name: str, shape: Tuple[int, ...], write_size: int = 1000, read_only: bool = False):
        assert os.path.isdir(path)

        full_path = os.path.join(path, name)
        if not os.path.isdir(full_path):
            os.mkdir(full_path)

        self.path = full_path
        self.tensor_file = os.path.join(full_path, self.TENSOR_FILE)
        self.index_file = os.path.join(full_path, self.INDEX_FILE)

        self.write_size = write_size
        self.read_only = read_only
        self._base_shape = shape

        print(f"Opening {self.tensor_file}")

        try:
            if self.read_only:
                f = h5py.File(self.tensor_file, 'r')
            else:
                f = h5py.File(self.tensor_file, 'a')
                if not self.DATASET_NAME in f:
                    f.create_dataset(
                        self.DATASET_NAME,
                        shape=(0, *shape),
                        maxshape=(None, *shape)
                    )
                if not os.path.isfile(self.index_file):
                    with open(self.index_file, 'wb') as file:
                        pickle.dump({}, file)
        except BlockingIOError as e:
            print(f"Unable to open {self.tensor_file}")
            raise e

        assert self.DATASET_NAME in f
        self._cache = f[self.DATASET_NAME]

        with open(self.index_file, 'rb') as file:
            self._index = pickle.load(file)

        self._new_cache = {}
        self._f = f

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self._index), *self._base_shape)

    def write_cache_to_disk(self):
        assert not self.read_only

        if len(self._new_cache) == 0:
            return

        # Write the freshly added data
        old_cache_size = self._cache.shape[0]
        new_cache_size = len(self._index)

        # Resize to fit the new data
        self._cache.resize((new_cache_size, *self._base_shape))

        # Write the new values
        for smile, value in self._new_cache.items():
            # Empty entry
            if not value is None:
                idx = self._index[smile]
                assert idx >= old_cache_size, f"{idx}, {old_cache_size}"
                self._cache[idx] = value

        # Update the index
        with open(self.index_file, 'wb') as f:
            pickle.dump(self._index, f)

        self._new_cache = {}

    def _add_to_cache(self, key: str, value: np.ndarray):
        assert not self.read_only
        assert not (key in self._index) and not (key in self._new_cache)

        self._index[key] = len(self._index)
        self._new_cache[key] = value

        # Write to disk when the data in memory is too much
        if len(self._index) - self._cache.shape[0] >= self.write_size:
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
            # Write everything to cache
            if not self.read_only:
                self.write_cache_to_disk()

            # Then look up
            ids = np.array([self._index[k] for k in key])

            # Make sure the ids are ordered (h5py supports only ordered access)
            if np.all(ids[:-1] < ids[1:]):
                values = self._cache[ids]

            # If the ids are not in order
            else:
                # Step 1: Sort the IDs and remember the original indices
                id_order = np.argsort(ids)
                sorted_ids = ids[id_order]

                # Step 2: Check for duplicate ids and remove them from the query
                duplicates = not np.all(sorted_ids[:-1] < sorted_ids[1:])
                if duplicates:
                    non_duplicate_mask = np.concatenate(
                        [np.array([True]).reshape(1), sorted_ids[:-1] < sorted_ids[1:]], 0
                    )
                    sorted_ids = sorted_ids[non_duplicate_mask]
                    id_lookup = np.cumsum(non_duplicate_mask) - 1
                else:
                    id_lookup = np.arange(id_order.shape[0])

                # Step 3: Access the array using the sorted IDs
                values = self._cache[sorted_ids]

                # Step 4: Sort the values back to the original order
                inverse_ids = np.empty_like(id_order)
                inverse_ids[id_order] = id_lookup
                values = values[inverse_ids]

            return values

    def __len__(self) -> int:
        return len(self._index)

    def __delitem__(self, key):
        raise NotImplementedError("Item deletion is not implemented")

    def __setitem__(self, key: str, value: Optional[np.ndarray]):
        assert not self.read_only
        assert value is None or value.shape == self._base_shape

        if not key in self:
            self._add_to_cache(key, value)
        elif key in self._new_cache:
            self._new_cache[key] = value
        else:
            idx = self._index[key]
            self._cache[idx] = value

    def __del__(self):
        self.close()

    def __bool__(self):
        return True

    def flush(self):
        if not self.read_only:
            self.write_cache_to_disk()

    def close(self):
        self.flush()
        print(f"Closing {self.tensor_file}")
        self._f.close()

    def keys(self):
        return self._index.keys()

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return [self[key] for key in self]
