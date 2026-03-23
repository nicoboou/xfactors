import torch
from torch.utils.data import Dataset


class Cars3DDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images, self.labels = images, labels
        self.transform = transform
        assert len(images) == len(labels), "Number of images and labels doesn't match"

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        if not isinstance(image, torch.Tensor):
            image = torch.Tensor(image)
        processed_img = (image.permute(2, 0, 1) / 255).to(torch.float32)
        # if self.transform: processed_img = self.transform(processed_img)
        return processed_img, self.labels[idx]
