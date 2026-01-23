import torch
from torch.utils.data import Dataset

class BloodMNISTDataset(Dataset) :

    def __init__(self, images, labels):
        self.images, self.labels = images, labels
        assert(len(images)==len(labels)), "Number of images and labels doesn't match" 

    def __len__(self) -> int :
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] :
        return self.images[idx], self.labels[idx]
