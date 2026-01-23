from .bloodmnist import BloodMNISTDataModule
from .shapes3d import Shapes3DDataModule
from .celeba import CelebADataModule
from .mnist  import MNISTDataModule
from .dsprites import DSpritesDataModule
from .latent import LatentDataModule
from .mpi3d import MPI3DDataModule
from .cars3d import Cars3DDataModule

__all__=["BloodMNISTDataModule", "MPI3DDataModule", "Shapes3DDataModule", "CelebADataModule", "MNISTDataModule", "DSpritesDataModule", "LatentDataModule",
         "Cars3DDataModule"]