from __future__ import annotations
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    """
    Wrapper autour torchvision.datasets.MNIST qui:
    - renvoie des images Tensor [C,H,W] (1x28x28 ou 3x28x28 si to_rgb=True)
    - renvoie des labels LongTensor de forme [1] (compat. labels[:,0])
    """
    def __init__(
        self,
        root: str,
        train: bool,
        to_rgb: bool = False,
        normalize: bool = False,
        download: bool = False,
    ) -> None:
        tfs = []

        # MNIST est déjà grayscale, mais si on veut 3 canaux:
        if to_rgb:
            # sur PIL -> 3 canaux, puis ToTensor => [3,28,28]
            tfs.append(transforms.Grayscale(num_output_channels=3))

        tfs.append(transforms.ToTensor())  # [C,28,28] en [0,1]

        if normalize:
            if to_rgb:
                # même stats répétées sur 3 canaux
                mean = (0.1307, 0.1307, 0.1307)
                std  = (0.3081, 0.3081, 0.3081)
            else:
                mean = (0.1307,)
                std  = (0.3081,)
            tfs.append(transforms.Normalize(mean, std))

        transform = transforms.Compose(tfs)

        # cible sous forme [1] (LongTensor), pour accéder à labels[:,0] downstream
        target_transform = transforms.Lambda(lambda y: torch.tensor([y], dtype=torch.long))

        self._ds = datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self._ds[idx]