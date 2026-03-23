import lightning as L
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

from os.path import join, exists

from disdiff_adapters.dataset import CelebADataset
from disdiff_adapters.utils.utils import load_h5, split
from disdiff_adapters.utils.const import CelebA


class CelebADataModule(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str = CelebA.Path.DATA,
        batch_size: int = 64,
        patch_size: tuple[int, list[int]] = (64, 64),
        num_workers: int = 4,
        pin_memory: bool = True,
        degradation_types: str | list[str] = "none",
        degradation_levels: list[int] | None = None,
        add_degradation_factor: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        if isinstance(degradation_types, str):
            self.degradation_types = [
                x.strip().lower() for x in degradation_types.split(",") if x.strip()
            ]
        else:
            self.degradation_types = [
                x.strip().lower() for x in degradation_types if x.strip()
            ]
        self.degradation_levels = (
            degradation_levels if degradation_levels is not None else [0, 1, 2, 3, 4, 5]
        )
        self.add_degradation_factor = add_degradation_factor

    def setup(self, stage: str | None = None) -> None:

        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ]
        )
        if stage in ("fit", None):
            self.train_dataset = CelebADataset(
                self.data_dir,
                split="train",
                transform=train_transforms,
                download=False,
                degradation_types=self.degradation_types,
                degradation_levels=self.degradation_levels,
                add_degradation_factor=self.add_degradation_factor,
            )

            # Replace CelebA with your dataset
            self.val_dataset = CelebADataset(
                self.data_dir,
                split="test",
                transform=val_transforms,
                download=False,
                degradation_types=self.degradation_types,
                degradation_levels=self.degradation_levels,
                add_degradation_factor=self.add_degradation_factor,
            )
        else:
            self.val_dataset = CelebADataset(
                self.data_dir,
                split="test",
                transform=val_transforms,
                download=False,
                degradation_types=self.degradation_types,
                degradation_levels=self.degradation_levels,
                add_degradation_factor=self.add_degradation_factor,
            )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> tuple[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> tuple[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
