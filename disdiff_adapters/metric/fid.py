import torch.nn as nn
from torch import rand
import torch
from torchmetrics.image.fid import FrechetInceptionDistance 
from lightning.pytorch.callbacks import Callback


class FID :
    
    def __init__(self, loader: torch.utils.data.DataLoader, gen_fn, feature: int=64, device="cuda:0") :
        self.device = device
        self.loader = loader
        self.gen_fn = gen_fn

        self.fid = FrechetInceptionDistance(feature=feature, normalize=True).to(self.device)

    def  __call__(self) :
        self.fid.reset()

        for batch in self.loader :
            images, labels = batch

            n_samples = len(batch[0])

            self.fid.update(images.to(self.device), real=True)
            self.fid.update(self.gen_fn(n_samples), real=False)

        return self.fid.compute().item()



class FIDCallback(Callback):
    def __init__(self, feature=64, every_n_epochs=1):
        super().__init__()

        self.feature  = feature
        self.every    = every_n_epochs       
        self._metric  = None                 


    def on_test_epoch_start(self, trainer, pl_module):
        gen_fn = lambda n: pl_module.generate(n) 
        self.loader = trainer.test_dataloaders
        self._metric = FID(loader=self.loader,
                           gen_fn=gen_fn,
                           feature=self.feature,
                           device=pl_module.device)

    def on_test_epoch_end(self, trainer, pl_module):
 
        if (trainer.current_epoch + 1) % self.every != 0:
            return
        fid_value = self._metric()
        pl_module.log("fid/test",
                      fid_value,
                      prog_bar=True,
                      sync_dist=True)
