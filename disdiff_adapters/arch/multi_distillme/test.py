import argparse
import torch
from lightning import LightningModule
import matplotlib.pyplot as plt
import glob
from os.path import join

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
import lightning as L

from disdiff_adapters.arch.vae import *
from disdiff_adapters.arch.multi_distillme import MultiDistillMeModule
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *
from disdiff_adapters.data_module import *
from disdiff_adapters.metric import *


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
        "--dataset",
        type=str,
        help="dataset name used.",
        default="bloodmnist"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        default=10
    )

    parser.add_argument(
        "--factor",
        type=int,
        default=0,
        help="Choose a factor to encode"
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
        "--loss_type",
        type=str,
        default="all",
        help="select loss type"
    )

    parser.add_argument(
        "--l_cov",
        type=float,
        default=1,
        help="use cross cov loss"
    )

    parser.add_argument(
        "--l_nce",
        type=float,
        default=1,
        help="use nce loss"
    )

    parser.add_argument(
        "--l_anti_nce",
        type=float,
        default=1,
        help="use anti nce loss"
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
        default="md"
    )
    return parser.parse_args()




def main(flags: argparse.Namespace) :

    torch.set_float32_matmul_precision('medium')
    warm_up = True if flags.warm_up == "True" else False

    res_block = ResidualBlock if flags.arch == "res" else SimpleConv

    print("\n\nYOU ARE LOADING A VAE\n\n")
    # Seed
    
    # Load data_module
    match flags.dataset:
        case "bloodmnist":
            data_module = BloodMNISTDataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 28

        case "shapes":
            data_module = Shapes3DDataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 64

        case "celeba":
            data_module = CelebADataModule(batch_size=flags.batch_size)
            in_channels = 3
            img_size = 64
            klw = 10e-4
            
        case _ :
            raise ValueError("Error flags.dataset")
        
    model_name="md"
    version=f"{model_name}_epoch={flags.max_epochs}_beta={(flags.beta_s,flags.beta_t)}_latent={(flags.latent_dim_t,flags.latent_dim_s)}_batch={flags.batch_size}_warm_up={warm_up}_lr={flags.lr}_arch={flags.arch}+l_cov={flags.l_cov}+l_nce={flags.l_nce}+l_anti_nce={flags.l_anti_nce}_{flags.key}" 
    ckpt_path = glob.glob(f"{LOG_DIR}/{model_name}_vf/{flags.dataset}/{version}/checkpoints/*.ckpt")[0]
    model = MultiDistillMeModule.load_from_checkpoint(ckpt_path)
    

    print(f"\nVERSION : {version}\n")

    trainer = Trainer(
            accelerator="auto",
            devices=flags.gpus,

            max_epochs=flags.max_epochs,
            log_every_n_steps=20,

            logger=TensorBoardLogger(
                save_dir=LOG_DIR+f"/{flags.version_model}",
                name=join(flags.dataset, f"loss_{flags.type_loss}", flags.experience),
                version=version,
                default_hp_metric=False,
            ),
            callbacks=[FIDCallback()]
        )
    
    trainer.test(model, data_module)
    

if __name__ == "__main__":
    flags = parse_args()
    main(flags)