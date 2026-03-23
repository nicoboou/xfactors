import argparse
import torch
import os
from os.path import join

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning import Trainer

from disdiff_adapters.arch.multi_distillme import Xfactors
from disdiff_adapters.arch.vae.block import SimpleConv, ResidualBlock
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *
from disdiff_adapters.data_module import *

SEED = 2025


def to_list(x: str) -> list[int]:
    return [int(elt) for elt in x.split(",")]


def to_list_float(x: str) -> list[float]:
    return [float(elt) for elt in x.split(",")]


def to_optional_list(x: str) -> list[int]:
    if x.strip() == "":
        return []
    return [int(elt) for elt in x.split(",")]


def list_to_str(l: list) -> str:
    s = ""
    for elt in l:
        s += str(elt)
        s += ","
    return s[:-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name used.", default="bloodmnist")
    parser.add_argument("--max_epochs", type=int, help="Max number of epochs.", default=50)
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--warm_up", type=str, default="False")
    parser.add_argument("--lr", type=float, default=10e-5, help="learning rate.")
    parser.add_argument("--beta_s", type=float, help="beta used", default=1.0)
    parser.add_argument("--beta_t", type=float, help="beta used", default=1.0)
    parser.add_argument("--latent_dim_s", type=int, help="dimension of the latent space", default=4)
    parser.add_argument(
        "--dims_by_factors",
        type=to_list,
        help="dimension of the latent space t",
        default="2",
    )
    parser.add_argument(
        "--select_factors",
        type=to_optional_list,
        default="",
        help="comma separated factor indices; empty = dataset defaults",
    )
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
        help="[none,combo,bilinear,bicubic,nearest_neighbor,blur,noise,jpeg]",
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
    res_block = ResidualBlock if flags.arch == "res" else SimpleConv

    match flags.dataset:
        case "dsprites":
            data_module = DSpritesDataModule(batch_size=flags.batch_size)
            param_class = DSprites
            in_channels = 1
            img_size = 64
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            default_select_factors = list(range(len(DSprites.Params.FACTORS_IN_ORDER)))
            select_factors = flags.select_factors if flags.select_factors else default_select_factors
            n = len(select_factors)
            dims_by_factors = n * [2]
            l_nce_by_factors = n * [(1 / n) * 0.1]
            binary_factor = False
            use_degradation_factor = False

        case "mpi3d":
            data_module = MPI3DDataModule(batch_size=flags.batch_size)
            param_class = MPI3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            default_select_factors = list(range(len(MPI3D.Params.FACTORS_IN_ORDER)))
            select_factors = flags.select_factors if flags.select_factors else default_select_factors
            n = len(select_factors)
            dims_by_factors = n * [2]
            l_nce_by_factors = n * [(1 / n) * 0.1]
            binary_factor = False
            use_degradation_factor = False

        case "shapes":
            param_class = Shapes3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            map_idx_labels = list(Shapes3D.Params.FACTORS_IN_ORDER)
            default_select_factors = list(range(len(map_idx_labels)))
            select_factors = flags.select_factors if flags.select_factors else default_select_factors
            degradation_factor_idx = map_idx_labels.index("degradation_level")
            use_degradation_factor = degradation_factor_idx in select_factors
            data_module = Shapes3DDataModule(
                batch_size=flags.batch_size,
                degradation_types=flags.degradation_types,
                degradation_levels=flags.degradation_levels,
                add_degradation_factor=use_degradation_factor,
            )
            n = len(select_factors)
            dims_by_factors = n * [2]
            l_nce_by_factors = n * [(1 / n) * 0.1]
            binary_factor = False

        case "celeba":
            param_class = CelebA
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = 1
            factor_value_1 = 0
            map_idx_labels = list(CelebA.Params.FACTORS_IN_ORDER)
            default_select_factors = list(CelebA.Params.REPRESENTANT_IDX)
            select_factors = flags.select_factors if flags.select_factors else default_select_factors
            degradation_factor_idx = map_idx_labels.index("degradation_level")
            use_degradation_factor = degradation_factor_idx in select_factors
            data_module = CelebADataModule(
                batch_size=flags.batch_size,
                degradation_types=flags.degradation_types,
                degradation_levels=flags.degradation_levels,
                add_degradation_factor=use_degradation_factor,
            )
            n = len(select_factors)
            dims_by_factors = n * [2]
            l_nce_by_factors = n * [(1 / n) * 0.1]
            binary_factor = not use_degradation_factor

        case "cars3d":
            data_module = Cars3DDataModule(batch_size=flags.batch_size)
            param_class = Cars3D
            in_channels = 3
            img_size = 64
            klw = 0.000001
            factor_value = -1
            factor_value_1 = -1
            default_select_factors = [1, 2]
            select_factors = flags.select_factors if flags.select_factors else default_select_factors
            n = len(select_factors)
            dims_by_factors = n * [2]
            l_nce_by_factors = n * [(1 / n) * 0.1]
            binary_factor = False
            use_degradation_factor = False

        case _:
            raise ValueError("Error flags.dataset")

    if flags.dataset not in {"celeba", "shapes"}:
        map_idx_labels = param_class.Params.FACTORS_IN_ORDER

    if any(f < 0 or f >= len(map_idx_labels) for f in select_factors):
        raise ValueError(f"Invalid select_factors={select_factors} for dataset={flags.dataset} with {len(map_idx_labels)} labels")

    print(f"dims_by_factors: {sum(dims_by_factors)}")

    model = Xfactors(
        in_channels=in_channels,
        img_size=img_size,
        latent_dim_s=flags.latent_dim_s,
        dims_by_factors=dims_by_factors,
        select_factors=select_factors,
        factor_value=factor_value,
        factor_value_1=factor_value_1,
        res_block=res_block,
        beta_s=flags.beta_s,
        beta_t=flags.beta_t,
        warm_up=warm_up,
        kl_weight=klw,
        type=flags.loss_type,
        l_cov=flags.l_cov,
        l_nce_by_factors=l_nce_by_factors,
        l_anti_nce=flags.l_anti_nce,
        map_idx_labels=map_idx_labels,
        temp=0.03,
        binary_factor=binary_factor,
    )

    model_name = "x"
    version = f"{model_name}_epoch={flags.max_epochs}_beta=({flags.beta_s},{flags.beta_t})_latent=({flags.latent_dim_s},{list_to_str(dims_by_factors)})_batch={flags.batch_size}_warm_up={warm_up}_lr={flags.lr}_arch={flags.arch}+l_cov={flags.l_cov}+l_nce={list_to_str(l_nce_by_factors)}+l_anti_nce={flags.l_anti_nce}+degradation={flags.degradation_types}_{flags.key}"
    print(f"\nVERSION : {version}\n")
    print(f"Select factor : {select_factors}, factor value : {factor_value}")

    local_log_dir = join(
        LOG_DIR,
        flags.version_model,
        flags.dataset,
        f"loss_{flags.loss_type}",
        flags.experience,
        version,
    )
    os.makedirs(local_log_dir, exist_ok=True)
    logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT", "disdiff_adapters"),
        name=version,
        save_dir=local_log_dir,
        group=join(flags.dataset, f"loss_{flags.loss_type}", flags.experience),
        log_model=False,
    )
    degradation_cfg = {
        "degradation_types": flags.degradation_types,
        "degradation_levels": flags.degradation_levels,
    }
    experiment = logger.experiment
    if hasattr(experiment, "config") and hasattr(experiment.config, "update"):
        experiment.config.update(degradation_cfg, allow_val_change=True)
    else:
        logger.log_hyperparams(degradation_cfg)

    ckpt_dir = os.path.join(local_log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:03d}",
    )

    trainer = Trainer(
        accelerator="auto",
        devices=flags.gpus,
        gradient_clip_val=3.0,
        max_epochs=flags.max_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[ckpt_cb, LearningRateMonitor("epoch")],
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    flags = parse_args()
    main(flags)
