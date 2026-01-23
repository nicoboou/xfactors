from __future__ import annotations
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from disdiff_adapters.utils import MNIST
from disdiff_adapters.dataset import MNISTDataset


class MNISTDataModule(LightningDataModule):
    """
    DataModule Lightning pour MNIST avec split train/val déterministe.
    - dims: (C,H,W) exposée pour modèles/summary
    - num_classes: 10
    """
    name: str = "mnist"

    def __init__(self, 
                 batch_size: int=64, 
                 num_workers: int=4, 
                 pin_memory: bool=True, 
                 val_ratio: float=0.1,
                 to_rgb: bool=False, 
                 normalize: bool=False, 
                 drop_last: bool=False):
        super().__init__()

        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

        self.num_classes = 10
        self.dims = (3 if to_rgb else 1, 28, 28)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio
        self.to_rgb = to_rgb
        self.normalize = normalize
        self.drop_last = drop_last

    def prepare_data(self) -> None:
        # download
        MNISTDataset(
            root=MNIST.Path.data_dir,
            train=True,
            to_rgb=self.to_rgb,
            normalize=self.normalize,
            download=True,
        )
        MNISTDataset(
            root=MNIST.Path.data_dir,
            train=False,
            to_rgb=self.to_rgb,
            normalize=self.normalize,
            download=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full_train = MNISTDataset(
                root=MNIST.Path.data_dir,
                train=True,
                to_rgb=self.to_rgb,
                normalize=self.normalize,
                download=False,
            )

            n_total = len(full_train)
            n_val = int(self.val_ratio * n_total)
            n_train = n_total - n_val

            # Split déterministe (seed fixe pour reproductibilité)
            self._train, self._val = random_split(
                full_train,
                lengths=[n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

        if stage in (None, "test"):
            self._test = MNISTDataset(
                root=MNIST.Path.data_dir,
                train=False,
                to_rgb=self.to_rgb,
                normalize=self.normalize,
                download=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
