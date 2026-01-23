from torchvision.datasets import CelebA

class CelebADataset(CelebA) :
    def _check_integrity(self):
        return True