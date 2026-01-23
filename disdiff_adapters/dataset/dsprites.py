import torch
from torch.utils.data import Dataset


class DSpritesDataset (Dataset) :

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images, self.labels = images, labels
        assert(len(images)==len(labels)), "Number of images and labels doesn't match" 

    def __len__(self) -> int :
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] :
        image = self.images[idx]
        labels = self.labels[idx]
        if type(image) != torch.Tensor : image = torch.Tensor(image)
        if type(labels) != torch.Tensor : labels = torch.Tensor(labels)
        processed_img = image.unsqueeze(0).to(torch.float32)
        return processed_img, labels
