# Deprecated file

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int, img_size: int, latent_dim: int, is_vae=True):
        """
        Load a variational encoder.

        Args:
            in_channels: int, channel of the original image
            img_size: int, for a 28x28px img_size=28
            latent_dim: int, size of a latent vector
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.out_encoder_shape = None
        self.is_vae=is_vae
        self.activation = nn.ELU

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), self.activation(), #/2
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), self.activation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), self.activation(), #/2
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), self.activation(),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1), self.activation(), #/2
        )

        # calcul automatique de la taille aplatie après convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            out = self.features(dummy)
            self.out_encoder_shape = out.shape[1:]
            self.flattened_size = out.shape[1]*out.shape[2]*out.shape[3]
        if is_vae:
            self.fc = nn.Sequential(nn.Flatten(),nn.Linear(self.flattened_size, latent_dim * 2))

    def forward(self, x):
        x = self.features(x)
        if self.is_vae :
            x = self.fc(x)
            mu, logvar = torch.chunk(x, 2, dim=1)
            return mu, logvar
        else : return x
