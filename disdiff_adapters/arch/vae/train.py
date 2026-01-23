import argparse
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
import lightning as L

from disdiff_adapters.arch.vae import *
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *
from disdiff_adapters.data_module import *

from os.path import join
from disdiff_adapters.utils import LOG_DIR

SEED = 2025

def to_list(x: str) -> list[str] :
    return [int(gpu_id) for gpu_id in x.split(",")]

def parse_args() -> argparse.Namespace:
    """
    epochs : max_epoch (int)
    loss : loss name (str)
    optim : optimizer (str)
    arch : model used (str)
    dataset : data module loaded (str)
    pretrained : is already loaded (bool)

    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Max number of epochs.",
        default=50,
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        help="beta used",
        default=1.0,
    )    

    
    parser.add_argument(
        "--latent_dim",
        type=int,
        help="dimension of the latent space",
        default=4
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name used.",
        default="bloodmnist"
    )

    parser.add_argument(
        "--is_vae",
        type=str,
        help="is a vae",
        default="True"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        default=10
    )

    parser.add_argument(
        "--warm_up",
        type=str,
        default="False"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=10e-5,
        help="learning rate."
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="def",
        help="Name of the architecture"
    )

    parser.add_argument(
        "--experience",
        type=str,
        default="",
        help="Name of the experience"
    )

    parser.add_argument(
        "--gpus",
        type=to_list,
        default="0",
        help="comma seperated list of gpus"
    )

    parser.add_argument(
        "--version_model",
        type=str,
        default="debug"
    )
    
    return parser.parse_args()

def main(flags: argparse.Namespace) :

    torch.set_float32_matmul_precision('medium')
    is_vae = True if flags.is_vae == "True" else False
    warm_up = True if flags.warm_up == "True" else False

    res_block = ResidualBlock if flags.arch == "res" else SimpleConv
    
    # Load data_module
    match flags.dataset:
        case "bloodmnist":
            data_module = BloodMNISTDataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 28
            klw = 0.0001

        case "shapes":
            data_module = Shapes3DDataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 64
            #klw = 0.000001
            klw = flags.batch_size/(3*1e5)
        
        case "celeba":
            data_module = CelebADataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 64
            klw = 0.000001
        case _ :
            raise ValueError("Error flags.dataset")


    if is_vae :
        print("\nVAE module\n") 
        model = VAEModule(in_channels = in_channels,
                    img_size=img_size,
                    latent_dim=flags.latent_dim,
                    res_block=res_block,
                    beta=flags.beta,
                    warm_up=warm_up,
                    kl_weights=klw,)
        model_name = "vae"
    else :
        print("\nAE module\n") 
        model = AEModule(in_channels = in_channels,
                    img_size=img_size,
                    latent_dim=flags.latent_dim)
        model_name = "ae"
    
    version=f"{model_name}_epoch={flags.max_epochs}_beta={flags.beta}_latent={flags.latent_dim}_warm_up={warm_up}_lr={flags.lr}_batch={flags.batch_size}_arch={flags.arch}"
    print(f"\nVERSION : {version}\n")

    trainer = Trainer(
            accelerator="auto",
            devices=flags.gpus,
            gradient_clip_val= 3.0,

            max_epochs=flags.max_epochs,
            log_every_n_steps=10,

            logger=TensorBoardLogger(
                save_dir=LOG_DIR+f"/{flags.version_model}",
                name=join(flags.dataset, flags.experience),
                version=version,
                default_hp_metric=False,
            ),
            callbacks=[
                ModelCheckpoint(monitor="loss/val", mode="min"),
                LearningRateMonitor("epoch"),
            ]
        )
    


    trainer.fit(model, data_module)


if __name__ == "__main__":
    flags = parse_args()
    main(flags)