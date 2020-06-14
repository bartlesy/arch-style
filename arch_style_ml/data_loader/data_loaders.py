from dataclasses import dataclass
from types import FunctionType
import cytoolz as toolz

from pathlib import Path
from typing import Union, List, Callable, Tuple, Any, Dict
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from arch_style_ml.data_loader.augmentation import (
    AugmentationFactoryBase,
    ArchStyleTransforms,
)

PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).
    """

    def __init__(self, dataset: torch.utils.data.Dataset, map_fn: FunctionType):
        self.dataset = dataset
        self.map_fn = map_fn

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        return self.map_fn(sample), target

    def __len__(self):
        return len(self.dataset)


class ArchStyleDataset(ImageFolder):
    def __init__(
        self,
        root: Union[str, Path] = PACKAGE_DIR,
        transformer: AugmentationFactoryBase = ArchStyleTransforms(),
        train_pct: float = 0.80,
        *args,
        **kwargs
    ) -> None:
        super().__init__(root=root, *args, **kwargs)
        self.transformer = transformer
        # train_n = int(len(self.samples) * train_pct)
        # test_n = len(self.samples) - train_n
        # train, test = torch.utils.data.random_split(self.samples, [train_n, test_n])
        train, test = train_test_split(
            self.samples, test_size=1 - train_pct, random_state=420
        )
        train_loader = toolz.compose(transformer.train_transform, self.loader)
        test_loader = toolz.compose(transformer.test_transform, self.loader)
        self.train_ds = MapDataset(train, train_loader)
        self.test_ds = MapDataset(test, test_loader)


class ArchStyleDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: ArchStyleDataset, *args, **kwargs):
        self.raw_ds = dataset
        self.args = args
        self.kwargs = kwargs
        super().__init__(dataset=self.raw_ds.train_ds, *args, **kwargs)

    def split_validation(self):
        return DataLoader(self.raw_ds.test_ds, *self.args, **self.kwargs)
