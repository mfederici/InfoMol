from copy import deepcopy

from encoder import CachedEncoder
from encoder.base import Encoder
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
            transform: Callable,
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        return self.transform(self.dataset[idx])



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

def prepare_molecular_dataset(
        dataset: Dataset,
        encoder: Encoder,
        representation_path: Optional[str] = None,
        read_only: bool = True,
)-> Dataset:
    smiles = dataset.smiles
    if hasattr(dataset, 'y'):
        y = dataset.y
    else:
        y = None

    if not (representation_path is None):
        encoder = CachedEncoder(
            encoder=encoder,
            cache_path=representation_path,
            read_only=read_only
        )

    transformed_dataset = TransformedDataset(
        smiles,
        encoder
    )

    if y is None:
        dataset = transformed_dataset
    else:
        dataset = JointTupleDataset(transformed_dataset, y)

    return dataset


def compute_mean(data: Dataset) -> Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]:
    mean = None
    count = None

    for entry in data:
        if mean is None:
            if isinstance(entry, tuple):
                entry = list(entry)
            mean = deepcopy(entry)
            if hasattr(mean, 'items'):
                count = {k: 0 for k in mean}
            elif isinstance(mean, list) or isinstance(mean, tuple):
                count = [0 for _ in mean]
            else:
                count = 0

        else:
            if hasattr(entry, 'items'):
                for k, v in entry.items():
                    if not isnan(v):
                        mean[k] += v
                        count[k] += 1
            elif isinstance(entry, list) or isinstance(entry, tuple):
                for i, v in enumerate(entry):
                    if not isnan(v):
                        mean[i] += v
                        count[i] += 1
            else:
                if not isnan(v):
                    mean += entry
                    count += 1
    if hasattr(mean, 'items'):
        for k, v in mean.items():
            mean[k] = mean[k] / float(count[k])
    elif isinstance(mean, list) or isinstance(mean, tuple):
        for i, v in enumerate(mean):
            mean[i] = mean[i] / float(count[i])
    else:
        mean = mean / float(count)

    return mean


def compute_std(
        data: Dataset,
        mean: Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]
) -> Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]:
    std = None
    count = None
    for entry in data:
        if std is None:
            if hasattr(entry, 'items'):
                std = {}
                count = {}
                for k, v in entry:
                    std[k] = (v - mean[k]) ** 2
                    count[k] = 0
            elif isinstance(entry, list) or isinstance(entry, tuple):
                std = []
                count = []
                for i, v in enumerate(entry):
                    std.append((v - mean[i]) ** 2)
                    count.append(0)
            else:
                std = (entry - mean) ** 2
                count = 0
        else:
            if hasattr(entry, 'items'):
                for k, v in entry.items():
                    if not isnan(v):
                        std[k] += (v - mean[k]) ** 2
                        count[k] += 1
            elif isinstance(entry, list) or isinstance(entry, tuple):
                for i, v in enumerate(entry):
                    if not isnan(v):
                        std[i] += (v - mean[i]) ** 2
                        count[i] += 1
            else:
                if not isnan(v):
                    std += (entry - mean) ** 2
                    count += 1

    if hasattr(std, 'items'):
        for k, v in std.items():
            mask = std[k] == 0
            std[k] = (std[k] / float(count[k] - 1)) ** 0.5

            if isinstance(mask, bool):
                if mask:
                    std[k] = 1.
            else:
                std[k][mask] = 1.

    elif isinstance(std, list) or isinstance(std, tuple):
        for i, v in enumerate(std):
            mask = std[i] == 0
            std[i] = (std[i] / float(count[i]-1)) ** 0.5
            if isinstance(mask, bool):
                if mask:
                    std[i] = 1.
            else:
                std[i][mask] = 1.
    else:
        mask = std == 0
        std = (std / float(count-1)) ** 0.5
        if isinstance(mask, bool):
            if mask:
                std = 1.
        else:
            std[mask] = 1.

    return std


class NormalizeEntry:
    def __init__(
            self,
            mean: Union[Dict[str, np.array], Tuple[float, ...], List[np.array]],
            std: Union[Dict[str, np.array], Tuple[float, ...], List[np.array]]
    ):
        self.mean = mean
        self.std = std

    def __call__(
            self,
            entry: Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]
    ) -> Union[Dict[str, np.array], Tuple[np.array, ...], List[np.array]]:
        if hasattr(entry, 'items'):
            norm_entry = {}
            for k, v in entry.items():
                norm_entry[k] = (v - self.mean[k]) / self.std[k]
        elif isinstance(entry, tuple) or isinstance(entry, list):
            norm_entry = []
            for i, v in enumerate(entry):
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
    mean = compute_mean(data)
    std = compute_std(data, mean)

    return TransformedDataset(
        data, NormalizeEntry(mean, std)
    )

class NormalizedDataset(TransformedDataset):
    def __init__(self, dataset: Dataset):
        mean = compute_mean(dataset)
        std = compute_std(dataset, mean)

        super().__init__(
            dataset=dataset,
            transform=NormalizeEntry(mean, std)
        )
