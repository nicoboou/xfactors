import argparse
import torch
import glob
from os.path import join
from lightning.pytorch.loggers import WandbLogger
from lightning import Trainer

from disdiff_adapters.arch.vae import *
from disdiff_adapters.arch.multi_distillme import Xfactors
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *
from disdiff_adapters.data_module import *
from disdiff_adapters.metric import *


def to_list(x: str) -> list[int]:
    return [int(elt) for elt in x.split(",")]


def to_list_float(x: str) -> list[float]:
    return [float(elt) for elt in x.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="dataset name used.", default="bloodmnist")

    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Max number of epochs.",
        default=50,
    )

    parser.add_argument("--batch_size", type=int, help="batch size", default=32)

    parser.add_argument("--warm_up", type=str, default="False")

    parser.add_argument("--lr", type=float, default=10e-5, help="learning rate.")

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

    parser.add_argument("--latent_dim_s", type=int, help="dimension of the latent space", default=4)

    parser.add_argument(
        "--dims_by_factors",
        type=to_list,
        help="dimension of the latent space t",
        default="2",
    )

    parser.add_argument("--select_factors", type=to_list, default="0", help="Choose a factor to encode")

    parser.add_argument("--factor_value", type=int, default=1, help="Choose a factor to encode")

    parser.add_argument("--arch", type=str, default="def", help="main value of the interest factor")

    parser.add_argument("--loss_type", type=str, default="all", help="select loss type")

    parser.add_argument("--l_cov", type=float, default=0, help="use cross cov loss")

    parser.add_argument("--l_nce_by_factors", type=to_list_float, default="1e-2", help="use nce loss")

    parser.add_argument("--l_anti_nce", type=float, default=0, help="use anti nce loss")

    parser.add_argument("--experience", type=str, default="", help="Name of the experience")

    parser.add_argument("--key", type=str, default="", help="key to add for the file")

    parser.add_argument("--gpus", type=to_list, default="0", help="comma seperated list of gpus")

    parser.add_argument("--version_model", type=str, default="debug")

    parser.add_argument(
        "--degradation_types",
        type=str,
        default="none",
        help="comma separated list: none,combo,bilinear,bicubic,nearest_neighbor,blur,noise,jpeg",
    )

    parser.add_argument(
        "--degradation_levels",
        type=to_list,
        default="0,1,2,3,4,5",
        help="comma separated degradation levels",
    )
    return parser.parse_args()


def main(flags: argparse.Namespace):

    torch.set_float32_matmul_precision("medium")
    warm_up = True if flags.warm_up == "True" else False

    use_degradation_factor = False
    if flags.dataset == "shapes":
        map_idx_labels = list(Shapes3D.Params.FACTORS_IN_ORDER)
        degradation_factor_idx = map_idx_labels.index("degradation_level") if "degradation_level" in map_idx_labels else len(map_idx_labels)
        use_degradation_factor = degradation_factor_idx in flags.select_factors
    elif flags.dataset == "celeba":
        map_idx_labels = list(CelebA.Params.FACTORS_IN_ORDER)
        degradation_factor_idx = map_idx_labels.index("degradation_level") if "degradation_level" in map_idx_labels else len(map_idx_labels)
        use_degradation_factor = degradation_factor_idx in flags.select_factors

    print("\n\nYOU ARE LOADING A VAE\n\n")
    # Seed

    # Load data_module
    match flags.dataset:
        case "bloodmnist":
            data_module = BloodMNISTDataModule(batch_size=flags.batch_size)

        case "shapes":
            data_module = Shapes3DDataModule(
                batch_size=flags.batch_size,
                degradation_types=flags.degradation_types,
                degradation_levels=flags.degradation_levels,
                add_degradation_factor=use_degradation_factor,
            )

        case "celeba":
            data_module = CelebADataModule(
                batch_size=flags.batch_size,
                degradation_types=flags.degradation_types,
                degradation_levels=flags.degradation_levels,
                add_degradation_factor=use_degradation_factor,
            )

        case _:
            raise ValueError("Error flags.dataset")

    model_name = "md"
    version = f"{model_name}_epoch={flags.max_epochs}_beta=({flags.beta_s},{flags.beta_t})_latent=({flags.latent_dim_s},{flags.dims_by_factors})_batch={flags.batch_size}_warm_up={warm_up}_lr={flags.lr}_arch={flags.arch}+l_cov={flags.l_cov}+l_nce={flags.l_nce_by_factors}+l_anti_nce={flags.l_anti_nce}_{flags.key}"
    ckpt_path = glob.glob(f"{LOG_DIR}/{model_name}_vf/{flags.dataset}/{version}/checkpoints/*.ckpt")[0]
    model = Xfactors.load_from_checkpoint(ckpt_path)

    print(f"\nVERSION : {version}\n")

    trainer = Trainer(
        accelerator="auto",
        devices=flags.gpus,
        max_epochs=flags.max_epochs,
        log_every_n_steps=20,
        logger=WandbLogger(
            project=os.getenv("WANDB_PROJECT", "disdiff_adapters"),
            name=version,
            save_dir=join(LOG_DIR, flags.version_model),
            group=join(flags.dataset, f"loss_{flags.loss_type}", flags.experience),
            log_model=False,
        ),
        callbacks=[FIDCallback()],
    )

    trainer.test(model, data_module)


if __name__ == "__main__":
    flags = parse_args()
    main(flags)
