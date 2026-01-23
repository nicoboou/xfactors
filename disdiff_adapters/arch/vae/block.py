import torch
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out
    
class SimpleConv(nn.Module) :
    def __init__ (self, in_channels: int, out_channels: int) :
        super(SimpleConv, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
                                nn.BatchNorm2d(out_channels), 
                                nn.LeakyReLU())
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, 
                 img_size: int, 
                 latent_dim: int, 
                 is_vae=True, 
                 activation: nn.Module=nn.LeakyReLU,
                 res_block: nn.Module=SimpleConv):
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
        self.activation = activation
        self.res_block = res_block

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(48), self.activation(), #/2
            self.res_block(48, 48),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(96), self.activation(), #/2
            self.res_block(96, 96),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(192), self.activation(), #/2
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), self.activation() #/1
        )

        # calcul automatique de la taille aplatie après convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            out = self.features(dummy)
            self.out_encoder_shape = out.shape[1:]
            self.flattened_size = out.shape[1]*out.shape[2]*out.shape[3]
        if is_vae:
            self.fc = nn.Sequential(nn.Flatten(),nn.Linear(self.flattened_size, latent_dim * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x = self.features(x)
        if self.is_vae :
            x = self.fc(x)
            mu, logvar = torch.chunk(x, 2, dim=1)
            return mu, logvar
        else : return x

class Decoder(nn.Module):
    """
    VAE Decoder.
    
    See VAE docstring.

    ConvTranspose2D double H,W with kernel=4, stride=2, padding=1
    """
    
    def __init__(self, out_channels: int, 
                 img_size: int, 
                 latent_dim: int, 
                 out_encoder_shape: tuple[int],
                 is_vae: bool=True,
                 activation: nn.Module=nn.LeakyReLU,
                 res_block: nn.Module=SimpleConv):
        """
        Load a decoder variational.

        Args:
            latent_dim: int, size of the input latent vector.
            out_shape: tuple[int], (C,H,W) shape of the output image
            out_encoder_shape: tuple[int], (C,H,W) shape of the last convolutional layer of the encoder.
        """

        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.out_encoder_shape = out_encoder_shape
        self.is_vae = is_vae
        self.activation = activation
        self.res_block = res_block

        C,H,W = self.out_encoder_shape

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, C*W*H), 
            nn.Unflatten(1, (C, W, H)),)
    
        self.net = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), #*1
            nn.BatchNorm2d(192),
            self.activation(),

            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1), #*2
            nn.BatchNorm2d(96),
            self.activation(),

            self.res_block(96, 96),

            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),#*2
            nn.BatchNorm2d(48),
            self.activation(),

            self.res_block(48, 48),

            nn.ConvTranspose2d(48, self.out_channels, kernel_size=4, stride=2, padding=1), #*2
            self.activation(self.out_channels, self.out_channels),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1), #*1
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vae : x = self.fc(x)

        x = self.net(x)
        if not (self.img_size > 0 and (self.img_size & (self.img_size - 1))) == 0 : 

            x: torch.Tensor = F.interpolate(x, 
                              size=(self.img_size, self.img_size), 
                              mode='bilinear', 
                              align_corners=False) #Padding
        activation = nn.Sigmoid()
        return activation(x)
