import lightning as L
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
import numpy as np

from os.path import join, exists

from disdiff_adapters.dataset.mpi3d import MPI3DDataset
from disdiff_adapters.utils.utils import load_h5, split
from disdiff_adapters.utils.const import MPI3D

class MPI3DDataModule(LightningDataModule) :
    
    def __init__(self, h5_path: str=MPI3D.Path.H5, 
                 train_path: str=MPI3D.Path.TRAIN,
                 val_path: str=MPI3D.Path.VAL,
                 test_path: str=MPI3D.Path.TEST,
                 ratio: int=0.8,
                 batch_size: int=8,
                 loader: DataLoader|None=None) :
        super().__init__()
        self.h5_path = h5_path
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path
        self.ratio = ratio
        self.batch_size = batch_size
        self.loader = loader

    def prepare_data(self, is_h5: bool=False):

        if not (exists(self.train_path) and exists(self.val_path) and exists(self.test_path)):
            if is_h5 :
                print("h5 file loading.")
                images, labels = load_h5(self.h5_path)
                train_images, train_labels, test_images, test_labels = split(images, labels)
                train_images, train_labels, val_images, val_labels = split(train_images, train_labels)
            else :
                data = np.load(MPI3D.Path.NPZ)
                train_images = data["train_images.npy"]
                train_labels = data["train_labels.npy"]
                test_images = data["test_images.npy"]
                test_labels = data["test_labels.npy"]

                train_images, train_labels, val_images, val_labels = split(train_images, train_labels)

            np.savez(self.train_path, images=train_images, labels=train_labels)
            np.savez(self.val_path, images=val_images, labels=val_labels)
            np.savez(self.test_path, images=test_images, labels=test_labels)

        else : pass

    def setup(self, stage: str|None) :
        if stage in ("fit", None) :
            train_images, train_labels = np.load(self.train_path)["images"],np.load(self.train_path)["labels"]
            val_images, val_labels = np.load(self.val_path)["images"],np.load(self.val_path)["labels"]

            print("load dataset - train")
            self.train_dataset = MPI3DDataset(train_images, train_labels)
            print("load dataset val")
            self.val_dataset = MPI3DDataset(val_images, val_labels)
        else :
            test_images, test_labels = np.load(self.test_path)["images"], np.load(self.test_path)["labels"]
            self.test_dataset = MPI3DDataset(test_images, test_labels)
        print("tensors loaded.")
        self.set_dataloader(self.loader)

    def train_dataloader(self):
        if self.loader is None : return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)
        else : return self.loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)
    
    def set_dataloader(self, loader: DataLoader|None) :
        self.loader = loader
