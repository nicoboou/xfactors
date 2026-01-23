import torch
from torch.utils.data import Dataset
from disdiff_adapters.utils.utils import load_h5
from disdiff_adapters.utils.const import Shapes3D

class Shapes3DDataset(Dataset) :

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images, self.labels = images, labels
        assert(len(images)==len(labels)), "Number of images and labels doesn't match" 

    def __len__(self) -> int :
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] :
        image = self.images[idx]
        if type(image) != torch.Tensor : image = torch.Tensor(image)
        processed_img = (image.permute(2,0,1)/255).to(torch.float32)
        return processed_img, self.labels[idx]
