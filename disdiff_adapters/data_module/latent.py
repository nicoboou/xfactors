import lightning as L
from lightning import LightningDataModule, LightningModule
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from os.path import join, exists

from disdiff_adapters.dataset import LatentDataset
from disdiff_adapters.arch.multi_distillme import MultiDistillMeModule
from disdiff_adapters.utils import *
from disdiff_adapters.data_module import *

class LatentDataModule(LightningDataModule) :
    
    def __init__(self, 
                ratio: int=0.8,
                batch_size: int=2**19,
                loader: DataLoader|None=None,
                Model_class: LightningModule=MultiDistillMeModule,
                data_name: str="shapes",
                ckpt_path: str="/projects/compures/alexandre/disdiff_adapters/disdiff_adapters/logs/md_with_val/shapes/loss_vae_nce/test_factor_floor/batch32/test_dim_s126/md_epoch=30_beta=(100.0,1.0)_latent=(126,2)_batch=32_warm_up=False_lr=1e-05_arch=res+l_cov=0.0+l_nce=0.1+l_anti_nce=0.0_/checkpoints/epoch=7-step=76800.ckpt",
                cond: str="both",
                pref_gpu: int=0,
                standard: bool=False,
                verbose: bool=True) :
        super().__init__()

        if data_name == "shapes" : self.Data_class = Shapes3D
        elif data_name == "dsprites" : self.Data_class = DSprites
        elif data_name == "celeba": self.Data_class = CelebA
        elif data_name == "mpi3d" : self.Data_class = MPI3D
        elif data_name == "cars3d" : self.Data_class = Cars3D
        else : raise ValueError


        h5_path=self.Data_class.Path.H5
        train_path=self.Data_class.Path.TRAIN
        val_path=self.Data_class.Path.VAL
        test_path=self.Data_class.Path.TEST

        self.h5_path = h5_path
        self.train_path, self.val_path, self.test_path = train_path, val_path, test_path

        self.data_name = data_name
        self.ratio = ratio
        self.batch_size = batch_size
        self.loader = loader
        self.ckpt_path = ckpt_path
        self.cond = cond
        self.Model_class = Model_class
        self.standard = standard
        device, _ = set_device(pref_gpu)
        self.md = self.Model_class.load_from_checkpoint(self.ckpt_path, map_location=device)
        self.verbose = verbose

    def prepare_data(self, is_h5: bool=False):
        if self.data_name != "celeba" : #CelebA is already splited
            if not (exists(self.train_path) and exists(self.val_path) and exists(self.test_path)):
                if is_h5 :
                    print("h5 file loading.")
                    images, labels = load_h5(self.h5_path)
                    train_images, train_labels, test_images, test_labels = split(images, labels)
                    train_images, train_labels, val_images, val_labels = split(train_images, train_labels)
                else :
                    data = np.load(self.Data_class.Path.NPZ)
                    train_images = data["train_images.npy"]
                    train_labels = data["train_labels.npy"]
                    test_images = data["test_images.npy"]
                    test_labels = data["test_labels.npy"]

                    train_images, train_labels, val_images, val_labels = split(train_images, train_labels)

                np.savez(self.train_path, images=train_images, labels=train_labels)
                np.savez(self.val_path, images=val_images, labels=val_labels)
                np.savez(self.test_path, images=test_images, labels=test_labels)

            else : pass
        else : pass

    def setup(self, stage: str|None) :

        if stage in ("fit", None) :

            train_images, train_labels = np.load(self.train_path)["images"],np.load(self.train_path)["labels"]
            val_images, val_labels = np.load(self.val_path)["images"],np.load(self.val_path)["labels"]

            print("load dataset - train")
            z_s_train, z_t_train = self._encode_all(train_images)
            z_s_train, z_t_train = self._standardize(z_s_train), self._standardize(z_t_train)
            self.train_dataset = LatentDataset(z_s_train, z_t_train, train_labels, cond="both")

            print("load dataset val")
            z_s_val, z_t_val = self._encode_all(val_images)
            z_s_val, z_t_val = self._standardize(z_s_val), self._standardize(z_t_val)
            self.val_dataset = LatentDataset(z_s_val, z_t_val, val_labels, cond="both")

        elif stage=="val":
            val_images, val_labels = np.load(self.val_path)["images"],np.load(self.val_path)["labels"]
            if self.verbose: print("load dataset val")
            z_s_val, z_t_val = self._encode_all(val_images)
            z_s_val, z_t_val = self._standardize(z_s_val), self._standardize(z_t_val)
            self.val_dataset = LatentDataset(z_s_val, z_t_val, val_labels, cond="both")
        else :
            if self.verbose: print("load test numpy.")
            test_images, test_labels = np.load(self.test_path)["images"], np.load(self.test_path)["labels"]
            if self.verbose: print("loaded test numpy.")
            z_s_test, z_t_test = self._encode_all(test_images)
            z_s_test, z_t_test = self._standardize(z_s_test), self._standardize(z_t_test)
            self.test_dataset = LatentDataset(z_s_test, z_t_test, test_labels, cond="both")

        if self.verbose: print("tensors loaded.")
        if self.loader is not None : self.set_dataloader(self.loader)
    
    def train_dataloader(self):
        if self.loader is None : return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)
        else : return self.loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)
    
    def set_dataloader(self, loader: DataLoader|None) :
        self.loader = loader


####################
    @torch.no_grad()
    def _encode_all(self, images, batch_size=2048) -> tuple[torch.Tensor, torch.Tensor] :
        self.md = self.md.eval()
        zs_list, zt_list = [], []
        images = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
        if isinstance(images, list) : images = torch.cat(images)
        if images.ndim == 3 : images = images.unsqueeze(3)
        assert images.ndim == 4, "Error, [B,C,H,W] format is required"
        if self.verbose: print("From images to latent vectors.")
        for i in tqdm(range(0, images.shape[0], batch_size)):
            x = images[i:i+batch_size].to(self.md.device, torch.float32).permute(0, 3, 1, 2)/images.max()
            mu_s, logvar_s = self.md.model.encoder_s(x)
            mu_t, logvar_t = self.md.model.encoder_t(x)
            zs_list.append(mu_s.detach().cpu())  
            zt_list.append(mu_t.detach().cpu())
        return torch.cat(zs_list, 0), torch.cat(zt_list, 0)

    def _standardize(self, z: torch.Tensor) -> torch.Tensor :
        if self.standard : return (z - z.mean(dim=0, keepdim=True))/(z.std(dim=0, keepdim=True, unbiased=False)+1e-8)
        else : return z 