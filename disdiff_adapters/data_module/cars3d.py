import lightning as L
from lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from os import remove
from os.path import join, exists
from shutil import rmtree

from disdiff_adapters.dataset import Cars3DDataset
from disdiff_adapters.utils.const import Cars3D

import torchvision.transforms as T

class Cars3DDataModule(LightningDataModule) :
    
    def __init__(self,
                train_path: str=Cars3D.Path.TRAIN,
                val_path: str=Cars3D.Path.VAL,
                test_path: str=Cars3D.Path.TEST,
                ratio: int=0.8,
                batch_size: int=8,
                loader: DataLoader|None=None,
                transform=None,) :
        
        super().__init__()

        self.transform = transform or T.Compose([
            T.ToPILImage(),    
            T.Resize((64, 64)),
            T.ToTensor(),            
        ])
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path
        self.ratio = ratio
        self.batch_size = batch_size
        self.loader = loader

    def prepare_data(self):

        if not (exists(self.train_path) and exists(self.val_path) and exists(self.test_path)):
            ds = load_dataset("randall-lab/cars3d", cache_dir=Cars3D.Path.CACHE, trust_remote_code=True)
            ds.save_to_disk(Cars3D.Path.LOCAL)
            #rmtree(Cars3D.Path.CACHE)

            ds = load_from_disk(Cars3D.Path.LOCAL)
            ds_tr = ds["train"]

            images = []
            labels = []
            for i in tqdm(range(17568)):
                image = torch.from_numpy(np.asarray(ds_tr[i]["image"]).copy())
                label = torch.from_numpy(np.asarray(ds_tr[i]["label"]).copy()).to(torch.long)
                images.append(image)
                labels.append(label)

            images_stacked = torch.stack(images)
            labels_stacked = torch.stack(labels)
            images_stacked = images_stacked.numpy()
            labels_stacked = labels_stacked.numpy()

            perm = torch.randperm(17568).numpy()
            images_stacked = images_stacked[perm]
            labels_stacked = labels_stacked[perm]

            images_tr = images_stacked[:15000]
            labels_tr = labels_stacked[:15000]

            images_val = images_stacked[15000:16500]
            labels_val = labels_stacked[15000:16500]

            images_te = images_stacked[16500:]
            labels_te = labels_stacked[16500:]

            np.savez(self.train_path, images=images_tr, labels=labels_tr)
            np.savez(self.val_path, images=images_val, labels=labels_val)
            np.savez(self.test_path, images=images_te, labels=labels_te)

        else : pass

    def setup(self, stage: str|None) :
        if stage in ("fit", None) :
            train_images, train_labels = np.load(self.train_path)["images"],np.load(self.train_path)["labels"]
            val_images, val_labels = np.load(self.val_path)["images"],np.load(self.val_path)["labels"]

            self.train_dataset = Cars3DDataset(train_images, train_labels, transform=self.transform)
            self.val_dataset   = Cars3DDataset(val_images, val_labels, transform=self.transform)
        else :
            test_images, test_labels = np.load(self.test_path)["images"], np.load(self.test_path)["labels"]
            self.test_dataset = Cars3DDataset(test_images, test_labels, transform=self.transform)
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
