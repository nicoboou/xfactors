## Deprecated file

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    VAE Decoder.

    See VAE docstring.

    ConvTranspose2D double H,W with kernel=4, stride=2, padding=1
    """

    def __init__(
        self,
        out_channels: int,
        img_size: int,
        latent_dim: int,
        out_encoder_shape: tuple[int],
        is_vae: bool = True,
    ):
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
        self.activation = nn.ELU
        C, H, W = self.out_encoder_shape

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, C * W * H),
            nn.Unflatten(1, (C, W, H)),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                256, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            self.activation(),  # *2
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            self.activation(),  # *1
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            self.activation(),  # *2
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            self.activation(),  # *1
            nn.ConvTranspose2d(
                32, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            self.activation(),  # *2
            nn.Conv2d(32, self.out_channels, kernel_size=3, padding=1, stride=1),  # *1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vae:
            x = self.fc(x)

        x = self.net(x)
        if not (self.img_size > 0 and (self.img_size & (self.img_size - 1))) == 0:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )  # Padding
        activation = nn.Sigmoid()
        return activation(x)
