import argparse
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt
import sys
import os
from os.path import join

from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
import lightning as L


os.chdir("/projects/compures/alexandre/disdiff_adapters")

sys.path.append("/projects/compures/alexandre/disdiff_adapters/")
print(sys.path)

 
from disdiff_adapters.arch.multi_distillme import MultiDistillMeModule
from disdiff_adapters.arch.vae.block import SimpleConv, ResidualBlock
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *
from disdiff_adapters.data_module import *

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
        "--dataset",
        type=str,
        help="dataset name used.",
        default="bloodmnist"
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Max number of epochs.",
        default=50,
    )
    
    parser.add_argument(
        "--beta_s",
        type=float,
        help="beta used",
        default=1.0,
    )    
    
    parser.add_argument(
        "--beta_t",
        type=float,
        help="beta used",
        default=1.0,
    )    
    
    parser.add_argument(
        "--latent_dim_s",
        type=int,
        help="dimension of the latent space",
        default=4
    )
    
    parser.add_argument(
        "--latent_dim_t",
        type=int,
        help="dimension of the latent space",
        default=4
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        default=32
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
        "--factor",
        type=int,
        default=0,
        help="Choose a factor to encode"
    )

    parser.add_argument(
        "--factor_value",
        type=int,
        default=1,
        help="Choose a factor to encode"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="def",
        help="main value of the interest factor"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="all",
        help="select loss type"
    )

    parser.add_argument(
        "--l_cov",
        type=float,
        default=0,
        help="use cross cov loss"
    )

    parser.add_argument(
        "--l_nce",
        type=float,
        default=1e-3,
        help="use nce loss"
    )

    parser.add_argument(
        "--l_anti_nce",
        type=float,
        default=0,
        help="use anti nce loss"
    )

    parser.add_argument(
        "--experience",
        type=str,
        default="",
        help="Name of the experience"
    )

    parser.add_argument(
        "--key",
        type=str,
        default="",
        help="key to add for the file"
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
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('medium')
    warm_up = True if flags.warm_up == "True" else False

    res_block = ResidualBlock if flags.arch == "res" else SimpleConv

    # Load data_module
    match flags.dataset:
        case "bloodmnist":
            data_module = BloodMNISTDataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 28
            klw = 0.0001
        
        case "dsprites":
            data_module = DSpritesDataModule(batch_size=flags.batch_size)
            param_class = DSprites
            in_channels = 1
            img_size = 64
            klw = 0.000001
            factor_value = -1
            select_factor = 0

        case "mpi3d":
            data_module = MPI3DDataModule(batch_size=flags.batch_size)
            param_class = MPI3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            select_factor = 1

        case "shapes":
            data_module = Shapes3DDataModule(batch_size=flags.batch_size)
            param_class = Shapes3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            select_factor = 0
            #klw = flags.batch_size/(3*1e5)
        
        case "celeba":
            data_module = CelebADataModule(batch_size=flags.batch_size)
            param_class = CelebA
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = 1
            select_factor = 26

        case "cars3d":
            data_module = Cars3DDataModule(batch_size=flags.batch_size)
            param_class = Cars3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            select_factor = 1

        case _ :
            raise ValueError("Error flags.dataset")
        
    map_idx_labels = param_class.Params.FACTORS_IN_ORDER

    model = MultiDistillMeModule(in_channels = in_channels,
                img_size=img_size,
                latent_dim_s=flags.latent_dim_s, 
                latent_dim_t=flags.latent_dim_t,
                select_factor=select_factor,
                factor_value=factor_value,
                res_block=res_block,
                beta_s=flags.beta_s,
                beta_t=flags.beta_t,
                warm_up=warm_up,
                kl_weight=klw,
                type=flags.loss_type,
                l_cov=flags.l_cov,
                l_nce=flags.l_nce,
                l_anti_nce=flags.l_anti_nce,
                map_idx_labels=map_idx_labels,
                temp=0.03)
    
    model_name = "md"
    
    version=f"{model_name}_epoch={flags.max_epochs}_beta=({flags.beta_s},{flags.beta_t})_latent=({flags.latent_dim_s},{flags.latent_dim_t})_batch={flags.batch_size}_warm_up={warm_up}_lr={flags.lr}_arch={flags.arch}+l_cov={flags.l_cov}+l_nce={flags.l_nce}+l_anti_nce={flags.l_anti_nce}_{flags.key}" 
    print(f"\nVERSION : {version}\n")
    print(f"Select factor : {select_factor}, factor value : {factor_value}")

    logger=TensorBoardLogger(
                save_dir=LOG_DIR+f"/{flags.version_model}",
                name=join(flags.dataset, f"loss_{flags.loss_type}", flags.experience),
                version=version,
                default_hp_metric=False,
            )

    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="loss/val",
        mode="min",
        save_top_k=1,          # garde uniquement le meilleur
        save_last=True,        # en plus, maintient checkpoints/last.ckpt (dernier)
        filename="best-{epoch:03d}",  # le nom du "best" (Lightning ajoutera la métrique)
    )

    trainer = Trainer(
        accelerator="auto",
        devices=flags.gpus,
        gradient_clip_val=3.0,
        max_epochs=flags.max_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,   # val à chaque époque => last.ckpt se met à jour à chaque epoch
        logger=logger,
        callbacks=[
            ckpt_cb,
            LearningRateMonitor("epoch"),
        ]
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    flags = parse_args()
    main(flags)