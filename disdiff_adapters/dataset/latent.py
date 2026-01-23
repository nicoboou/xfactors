import torch
from torch.utils.data import Dataset
from lightning import LightningModule
from disdiff_adapters.arch.multi_distillme import *

class LatentDataset(Dataset) :

    def __init__(self, 
                 z_s: torch.Tensor, 
                 z_t: torch.Tensor,
                 labels: torch.Tensor, 
                 cond: str="t") :
        super().__init__()

        assert cond in ["s", "t", "both", "cat"], "cond error"
        assert z_s.shape[0] == z_t.shape[0] == labels.shape[0], "images and labels should have the same shape[0]"
        if not isinstance(z_s, torch.Tensor) : z_s = torch.tensor(z_s, dtype=torch.float32)
        if not isinstance(z_s, torch.Tensor) : z_t = torch.tensor(z_t, dtyp=torch.float32)
        if not isinstance(labels, torch.Tensor) : labels = torch.tensor(labels, dtype=torch.long)
        
        self.cond = cond
        self.labels = labels
        self.z_s = z_s
        self.z_t = z_t
    
    def __len__(self) -> int :
        return self.z_s.shape[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.cond == "t" : z_returned = self.z_s[index]
        elif self.cond == "s" : z_returned = self.z_t[index]
        elif self.cond == "both" : return self.z_s[index], self.z_t[index], self.labels[index]
        else : z_returned =  torch.cat([self.z_s[index], self.z_t[index]], dim=0)
        return z_returned, self.labels[index] #[C, H, W]