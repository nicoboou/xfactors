import lightning as L
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
import numpy as np

from os.path import join, exists

from disdiff_adapters.dataset.shapes3d import Shapes3DDataset
from disdiff_adapters.utils.utils import load_h5, split
from disdiff_adapters.utils.const import Shapes3D


class Shapes3DDataModule(LightningDataModule):
    def __init__(
        self,
        h5_path: str = Shapes3D.Path.H5,
        train_path: str = Shapes3D.Path.TRAIN,
        val_path: str = Shapes3D.Path.VAL,
        test_path: str = Shapes3D.Path.TEST,
        ratio: int = 0.8,
        batch_size: int = 8,
        loader: DataLoader | None = None,
        degradation_types: str | list[str] = "none",
        degradation_levels: list[int] | None = None,
        add_degradation_factor: bool = False,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path
        self.ratio = ratio
        self.batch_size = batch_size
        self.loader = loader
        if isinstance(degradation_types, str):
            self.degradation_types = [x.strip().lower() for x in degradation_types.split(",") if x.strip()]
        else:
            self.degradation_types = [x.strip().lower() for x in degradation_types if x.strip()]
        self.degradation_levels = degradation_levels if degradation_levels is not None else [0, 1, 2, 3, 4, 5]
        self.add_degradation_factor = add_degradation_factor

    def prepare_data(self, is_h5: bool = False):

        if not (exists(self.train_path) and exists(self.val_path) and exists(self.test_path)):
            if is_h5:
                print("h5 file loading.")
                images, labels = load_h5(self.h5_path)
                train_images, train_labels, test_images, test_labels = split(images, labels)
                train_images, train_labels, val_images, val_labels = split(train_images, train_labels)
            else:
                data = np.load(Shapes3D.Path.NPZ)
                train_images = data["train_images.npy"]
                train_labels = data["train_labels.npy"]
                test_images = data["test_images.npy"]
                test_labels = data["test_labels.npy"]

                train_images, train_labels, val_images, val_labels = split(train_images, train_labels)

            np.savez(self.train_path, images=train_images, labels=train_labels)
            np.savez(self.val_path, images=val_images, labels=val_labels)
            np.savez(self.test_path, images=test_images, labels=test_labels)

        else:
            pass

    def setup(self, stage: str | None):
        if stage in ("fit", None):
            train_images, train_labels = np.load(self.train_path)["images"], np.load(self.train_path)["labels"]
            val_images, val_labels = np.load(self.val_path)["images"], np.load(self.val_path)["labels"]

            print("load dataset - train")
            self.train_dataset = Shapes3DDataset(
                train_images,
                train_labels,
                degradation_types=self.degradation_types,
                degradation_levels=self.degradation_levels,
                add_degradation_factor=self.add_degradation_factor,
            )
            print("load dataset val")
            self.val_dataset = Shapes3DDataset(
                val_images,
                val_labels,
                degradation_types=self.degradation_types,
                degradation_levels=self.degradation_levels,
                add_degradation_factor=self.add_degradation_factor,
            )
        elif stage == "val":
            val_images, val_labels = np.load(self.val_path)["images"], np.load(self.val_path)["labels"]
            print("load dataset val")
            self.val_dataset = Shapes3DDataset(
                val_images,
                val_labels,
                degradation_types=self.degradation_types,
                degradation_levels=self.degradation_levels,
                add_degradation_factor=self.add_degradation_factor,
            )

        else:
            test_images, test_labels = np.load(self.test_path)["images"], np.load(self.test_path)["labels"]
            self.test_dataset = Shapes3DDataset(
                test_images,
                test_labels,
                degradation_types=self.degradation_types,
                degradation_levels=self.degradation_levels,
                add_degradation_factor=self.add_degradation_factor,
            )
        print("tensors loaded.")
        self.set_dataloader(self.loader)

    def train_dataloader(self):
        if self.loader is None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)
        else:
            return self.loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)

    def set_dataloader(self, loader: DataLoader | None):
        self.loader = loader
