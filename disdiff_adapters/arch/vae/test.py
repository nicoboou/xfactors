import argparse
import glob
import os

from lightning.pytorch.loggers import WandbLogger
from lightning import Trainer

from disdiff_adapters.arch.vae import *
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *
from disdiff_adapters.data_module import *
from disdiff_adapters.metric import *


def to_list(x: str) -> list[str]:
    return [int(gpu_id) for gpu_id in x.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, help="Max number of epochs.", default=50)
    parser.add_argument("--beta", type=float, help="beta used", default=1.0)
    parser.add_argument("--latent_dim", type=int, help="dimension of the latent space", default=4)
    parser.add_argument("--dataset", type=str, help="dataset name used.", default="bloodmnist")
    parser.add_argument("--batch_size", type=int, help="batch size", default=10)
    parser.add_argument("--warm_up", type=str, default="False")
    parser.add_argument("--lr", type=float, default=10e-5, help="learning rate.")
    parser.add_argument("--arch", type=str, default="def", help="Name of the architecture")
    parser.add_argument("--gpus", type=to_list, default=["0"], help="comma seperated list of gpus")
    parser.add_argument("--experience", type=str, default="", help="Name of the experience")
    parser.add_argument("--gpus", type=to_list, default="0", help="comma seperated list of gpus")
    parser.add_argument("--version_model", type=str, default="debug")
    return parser.parse_args()


def main(flags: argparse.Namespace):
    is_vae = True if flags.is_vae == "True" else False
    warm_up = True if flags.warm_up == "True" else False

    print("\n\nYOU ARE LOADING A VAE\n\n")
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

        case _:
            raise ValueError("Error flags.dataset")

    callbacks = []

    if is_vae:
        model_class = VAEModule
        model_name = "vae"
        callbacks.append(FIDCallback())
    else:
        model_class = AEModule
        model_name = "ae"

    version = f"{model_name}_epoch={flags.max_epochs}_beta={flags.beta}_latent={flags.latent_dim}_warm_up={warm_up}_lr={flags.lr}_batch={flags.batch_size}_arch={flags.arch}"
    print(f"\nVERSION : {version}\n")

    ckpt_path = glob.glob(f"{LOG_DIR}/{flags.version_model}/{flags.dataset}/{flags.experience}/{version}/checkpoints/*.ckpt")[0]
    model = model_class.load_from_checkpoint(ckpt_path)

    print(f"\nVERSION : {version}\n")

    trainer = Trainer(
        accelerator="auto",
        devices=flags.gpus,
        max_epochs=flags.max_epochs,
        log_every_n_steps=20,
        logger=WandbLogger(
            project=os.getenv("WANDB_PROJECT", "disdiff_adapters"),
            name=version,
            save_dir=join(LOG_DIR, model_name),
            group=flags.dataset,
            log_model=False,
        ),
        callbacks=callbacks,
    )

    trainer.test(model, data_module)


if __name__ == "__main__":
    flags = parse_args()
    main(flags)
