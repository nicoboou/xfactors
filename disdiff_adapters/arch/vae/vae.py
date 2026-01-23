import torch
import torch.nn as nn
from lightning import LightningModule
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

from disdiff_adapters.arch.vae import Encoder, Decoder, ResidualBlock, SimpleConv
from disdiff_adapters.utils import sample_from, display
from disdiff_adapters.loss import *


class _VAE(torch.nn.Module) :
    def __init__(self, in_channels: int, 
                 img_size: int, 
                 latent_dim: int, 
                 activation: nn.Module=nn.LeakyReLU,
                 res_block: nn.Module=SimpleConv):
        
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim,
                               activation=activation,
                               res_block=res_block)
        
        self.decoder = Decoder(out_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim,
                               out_encoder_shape=self.encoder.out_encoder_shape,
                               activation=activation,
                               res_block=res_block)
        
    def forward(self, images: torch.Tensor, test: bool=False) :
        mus_logvars = self.encoder(images)
        z = sample_from(mus_logvars, test)
        image_hat_logits = self.decoder(z)

        return image_hat_logits, mus_logvars

class VAEModule(LightningModule) :

    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim: int,
                 activation: nn.Module=nn.LeakyReLU,
                 res_block: nn.Module=SimpleConv,
                 beta: float=1.0,
                 warm_up: bool=False,
                 kl_weights: float= 10e-4,
                 lr: float=10**(-5)) :
        
        super().__init__()
        self.save_hyperparameters()

        self.model = _VAE(in_channels=self.hparams.in_channels,
                          img_size=self.hparams.img_size,
                          latent_dim=self.hparams.latent_dim,
                          activation=self.hparams.activation,
                          res_block=self.hparams.res_block)
        self.images_test_buff = None
        self.images_train_buff = None
    
            
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0)

        return {"optimizer": optim, 
                "lr_scheduler": {
                                "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer=optim,gamma=0.95),
                                "monitor" :"loss/val",
                                "interval": "epoch",      
                                "frequency": 1,
                                }
                }
    
    def generate(self, nb_samples: int=8) -> torch.Tensor :
        eps = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim])).to(self.device, torch.float32)

        x_hat_logits = self.model.decoder(eps)
        return x_hat_logits
    
    def show_reconstruct(self, images: torch.Tensor) :
        images = images.to(self.device)[:8]
        images_gen, _ = self(images, test=True)

        fig, axes = plt.subplots(len(images), 2, figsize=(7, 20))

        for i in range(len(images)) :
            img = images[i]
            img_gen = images_gen[i]

            images_proc = (255*((img - img.min()) / (img.max() - img.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()
            images_gen_proc = (255*((img_gen - img_gen.min()) / (img_gen.max() - img_gen.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()

            axes[i,0].imshow(images_proc)
            axes[i,1].imshow(images_gen_proc)

            axes[i,0].set_title("original")
            axes[i,1].set_title("reco")
        plt.tight_layout()
        plt.show()

    
    def forward(self, images: torch.Tensor, test: bool=False) -> tuple[torch.Tensor]:
        image_hat_logits, mus_logvars= self.model(images, test)

        return image_hat_logits, mus_logvars
    
    def loss(self, image_hat_logits, mus_logvars, images, log_components=False) -> float :
        max_beta = self.hparams.beta
        mus, logvars = mus_logvars

        # beta warm-up
        if self.hparams.warm_up :
            start_epoch = int(self.trainer.max_epochs*1/5)
            epoch_limit = int(self.trainer.max_epochs*2/5)

            if self.current_epoch < start_epoch:
                beta = 0.0
            elif self.current_epoch <= epoch_limit:
                progress = (self.current_epoch - start_epoch) / (epoch_limit - start_epoch)
                beta = max_beta * progress
            else:
                beta = max_beta
        else : beta=max_beta
        
        weighted_kl = beta * self.hparams.kl_weights * kl(mus, logvars)

        reco = mse(image_hat_logits, images)

        if log_components :
            self.log("loss/kl", weighted_kl)
            self.log("loss/reco", reco)
            self.log("loss/beta", beta)

        return weighted_kl+reco
    
    def training_step(self, batch: tuple[torch.Tensor]) -> float:
        images, labels = batch
        image_hat_logits,mus_logvars = self.forward(images)
        loss = self.loss(image_hat_logits, mus_logvars, images, log_components=True)

        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        self.log("loss/train", loss)
        if self.images_train_buff is None : self.images_train_buff = images
        return loss

    def validation_step(self, batch: tuple[torch.Tensor]):
        images, labels = batch
        image_hat_logits, mus_logvars = self.forward(images)
        loss = self.loss(image_hat_logits, mus_logvars, images)

        self.log("loss/val", loss, sync_dist=True)
        print(f"Val loss: {loss}")
    
    def test_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        image_hat_logits, mus_logvars = self.forward(images, test=True)

        weighted_kl= self.hparams.beta*kl(*mus_logvars)
        reco = mse(image_hat_logits, images)

        self.log("loss/reco_test", reco, sync_dist=True)
        self.log("loss/kl_test", weighted_kl, sync_dist=True)
        self.log("loss/test", reco+weighted_kl, sync_dist=True)

        if self.images_test_buff is None : self.images_test_buff = images

    def on_train_epoch_end(self):
        epoch = self.current_epoch

        if epoch % 1 == 0:
            self.show_reconstruct(self.images_train_buff)

            try : os.mkdir(os.path.join(self.logger.log_dir, f"epoch_{epoch}"))
            except FileExistsError as e : pass

            save_reco_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"reco_{epoch}.png")
            plt.gcf().savefig(save_reco_path)

            images_gen = self.generate()
            save_gen_path = os.path.join(self.logger.log_dir, f"epoch_{epoch}", f"gen_{epoch}.png")
            vutils.save_image(images_gen.detach().cpu(), save_gen_path)

        

    def on_test_end(self):
        images_gen = self.generate()
        labels_gen = torch.zeros([images_gen.shape[0],1])

        display((images_gen.detach().cpu(), labels_gen.detach().to("cpu")))

        self.logger.experiment.add_figure("img/gen", plt.gcf())

        self.show_reconstruct(self.images_test_buff)
        self.logger.experiment.add_figure("img/reco", plt.gcf())

        vutils.save_image(images_gen.detach().cpu(), os.path.join(self.logger.log_dir, "gen.png"))





