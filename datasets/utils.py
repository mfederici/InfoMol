from collections import defaultdict
from copy import deepcopy
from torch_geometric.datasets import MoleculeNet

from encoder.base import Encoder, DEEP_ENCODER
from typing import Union, Any, Tuple, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = x - self.mean
        x = x / self.std
        return x


class TransformedDataset(Dataset):
    def __init__(
            self,
            dataset: Union[Dataset, np.ndarray, torch.Tensor],
            transform: Union[Callable, Dict[str, Callable], List[Callable]],
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        original_data = self.dataset[idx]

        if isinstance(self.transform, dict):
            transformed_data = {key: transform(original_data) for key, transform in self.transform.items()}
        elif isinstance(self.transform, list):
            transformed_data = tuple([transform(original_data) for transform in self.transform])
        else:
            transformed_data = self.transform(original_data)

        return transformed_data


class JointTupleDataset(Dataset):
    def __init__(self, *datasets: Tuple[Union[Dataset, np.ndarray, torch.Tensor], ...]):
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("Please provide at least one dataset")

        l = len(datasets[0])
        for dataset in datasets:
            if len(dataset) != l:
                raise ValueError(f"All datasets need to have the same lenght: {l}!={len(dataset)}")

        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, item: Union[int, slice]) -> Tuple[Any, ...]:
        return tuple([dataset[item] for dataset in self.datasets])

    def __del__(self):
        for dataset in self.datasets:
            del dataset




def compute_mean_std(data: Dataset) -> Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]:
    values = None

    for entry in data:
        if values is None:
            if hasattr(entry, 'items'):
                values = defaultdict(list)
            elif isinstance(entry, list) or isinstance(entry, tuple):
                values = []
                for v in entry:
                    assert (
                        isinstance(v, float) or
                        isinstance(v, int) or
                        isinstance(v, np.ndarray) or
                        torch.is_tensor(v)
                    )
                    values.append(v)
            else:
                assert (
                    isinstance(entry, float) or
                    isinstance(entry, int) or
                    isinstance(entry, np.ndarray) or
                    torch.is_tensor(entry)
                )
                values = []

        if hasattr(values, 'items'):
            for k, v in entry.items():
                if not isnan(v):
                    values[k].append(v.reshape(1,-1))
        elif isinstance(entry, list) or isinstance(entry, tuple):
            for i, v in enumerate(entry):
                if not isnan(v):
                    values[i].append(v.reshape(1,-1))
        else:
            if not isnan(entry):
                values.append(entry.reshape(1,-1))


    if hasattr(values, 'items'):
        mean = {}
        std = {}
        for k, v in values.items():
            vs = np.concatenate(values[k])
            mean[k] = vs.mean(0)
            std[k] = vs.std(0)

    elif isinstance(values, list):
        mean = []
        std = []
        for i, v in enumerate(values):
            vs = np.concatenate(values[i])
            mean.append(vs.mean(0))
            std.append(vs.std(0))
    else:
        vs = np.concatenate(values)
        mean = vs.mean(0)
        std = vs.std(0)

    return mean, std


class NormalizeEntry:
    def __init__(
            self,
            mean: Union[Dict[str, np.array], Tuple[float, ...], List[np.array]],
            std: Union[Dict[str, np.array], Tuple[float, ...], List[np.array]],
            min_std: float = 1e-6
    ):
        self.mean = mean
        if hasattr(std, 'items'):
            for k, v in std.items():
                v[np.abs(v) < min_std] = 1
                std[k] = v
        elif isinstance(std, list):
            for i, v in enumerate(std):
                v[np.abs(v) < min_std] = 1
                std[i] = v
        else:
            if np.abs(std)<min_std:
                std = 1

        self.std = std


    def __call__(
            self,
            entry: Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]
    ) -> Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]:
        if hasattr(entry, 'items'):
            norm_entry = {}
            for k in entry:
                if k in self.mean:
                    norm_entry[k] = (entry[k] - self.mean[k]) / self.std[k]
                else:
                    norm_entry[k] = entry[k]

        elif isinstance(self.mean, tuple) or isinstance(entry, list):
            norm_entry = []
            assert len(entry) == len(self.mean)
            for i, v in enumerate(self.mean):
                norm_entry.append((v - self.mean[i]) / self.std[i])
        else:
            norm_entry = (entry - self.mean) / self.std

        return norm_entry

def isnan(x):
    if isinstance(x, np.ndarray):
        return np.isnan(np.sum(x))
    elif torch.is_tensor(x):
        return torch.isnan(torch.sum(x))
    else:
        return False


def normalize_data(
        data: Dataset,
) -> Dataset:
    mean, std = compute_mean_std(data)


    return TransformedDataset(
        data, NormalizeEntry(mean, std)
    )

class NormalizedDataset(TransformedDataset):
    def __init__(self, dataset: Dataset):
        mean, std = compute_mean_std(dataset)
        super().__init__(
            dataset=dataset,
            transform=NormalizeEntry(mean, std)
        )


def prepare_molecular_dataset(
        dataset: Dataset,
        encoder: Union[Dict[str, Encoder], Encoder],
)-> Dataset:
    if isinstance(encoder, Encoder):
        encoder = {'x': encoder}

    if isinstance(dataset, MoleculeNet):
        dataset = JointTupleDataset(dataset.smiles, dataset.y)

    data = dataset[0]

    if isinstance(data, list) or isinstance(data, tuple):
        assert isinstance(data[0], str)
        def encode_smiles(args) -> Dict[str, Any]:
            nonlocal encoder
            smiles = args[0]
            args = args[1:]
            data = {}
                # 'smiles': smiles,
            # }
            for name, e in encoder.items():
                data[name] = e(smiles)

            if len(args) == 1:
                data['y'] = args[0]
            elif len(args)>1:
                for i, arg in enumerate(args):
                    data[f'target_{i+1}'] = arg
            return data
        wrapped_encoder = encode_smiles
    elif isinstance(data, dict):
        assert 'smiles' in data
        for key in data:
            assert not (key in encoder)
        def encode_smiles(data_dict: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal encoder
            smiles = data_dict['smiles']
            del data_dict['smiles']
            data_dict.update({name: e(smiles) for name, e in encoder.items()})
            return data_dict
        wrapped_encoder = encode_smiles
    else:
        wrapped_encoder = encoder

    dataset = TransformedDataset(
        dataset,
        wrapped_encoder
    )

    normalization = None

    # Normalize the dataset for deep encoders
    if isinstance(encoder, Encoder):
        if encoder.type == DEEP_ENCODER:
            mean, std = compute_mean_std(dataset)
            normalization = NormalizeEntry({'x': mean['x']}, {'x': std['x']})
    else:
        keys_to_normalize = set()
        for key, e in encoder.items():
            if e.type == DEEP_ENCODER:
                keys_to_normalize.add(key)
        mean, std = compute_mean_std(dataset)
        mean = {k:v for k,v in mean.items() if k in keys_to_normalize}
        std = {k:v for k,v in std.items() if k in keys_to_normalize}
        normalization = NormalizeEntry(mean, std)

    if normalization:
        dataset = TransformedDataset(dataset, transform=normalization)

    return dataset

