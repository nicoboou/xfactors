import torch
from lightning import LightningModule
import matplotlib.pyplot as plt

from disdiff_adapters.arch.vae import *
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *

class _AE(torch.nn.Module) :
    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim: int) :
        
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim,
                               is_vae=False)
        
        self.decoder = Decoder(out_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim,
                               out_encoder_shape=self.encoder.out_encoder_shape,
                               is_vae=False)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor :
        z = self.encoder(images)
        image_hat_logits = self.decoder(z)

        return image_hat_logits

class AEModule(LightningModule) :

    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim: int,
                 beta: float=1.0,) :
        
        super().__init__()
        self.save_hyperparameters()

        self.model = _AE(in_channels=self.hparams.in_channels,
                          img_size=self.hparams.img_size,
                          latent_dim=self.hparams.latent_dim)
        self.images_buff = None
    
            
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=6e-5, weight_decay=1e-2)
    
    def show_reconstruct(self, images: torch.Tensor) :
        images = images.to(self.device)
        images_gen = self(images)

        fig, axes = plt.subplots(len(images), 2, figsize=(7, 20))

        for i in range(len(images)) :
            images_proc = (images[i]*255).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()
            images_gen_proc = (images_gen[i]*255).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()

            axes[i,0].imshow(images_proc)
            axes[i,1].imshow(images_gen_proc)

            axes[i,0].set_title("original")
            axes[i,1].set_title("reco")
        plt.show()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_hat_logits= self.model(images)

        return image_hat_logits
    
    def loss(self, image_hat_logits, images, log_components=False) -> float :
        reco = mse(image_hat_logits, images)

        if log_components :

            self.log("loss/reco", reco)

        return reco
    
    def training_step(self, batch: tuple[torch.Tensor]) -> float:
        images, labels = batch
        image_hat_logits= self.forward(images)
        loss = self.loss(image_hat_logits, images, log_components=True)

        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor]):
        images, labels = batch
        image_hat_logits  = self.forward(images)
        loss = self.loss(image_hat_logits, images)

        self.log("loss/val", loss)
        print(f"Val loss: {loss}")
    
    def test_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        image_hat_logits = self.forward(images)


        reco = mse(image_hat_logits, images)

        self.log("loss/test", reco)

        if self.images_buff is None : self.images_buff = images


    def on_test_end(self):

        self.show_reconstruct(self.images_buff)
        self.logger.experiment.add_figure("img/reco", plt.gcf())        




