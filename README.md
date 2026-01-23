# XFactors

## Project description
XFactors is a method for partial or complete latent space disentanglement. Our architecture enables the supervised disentanglement of image factors $(f_1, \ldots, f_k)$ and allows for the selection of specific directions in the latent space where each factor will be encoded.

It is also possible to encode only a subset of factors (e.g., CelebA, which is annotated with 40 factors; one can encode only specific factors of interest) without omitting the encoding of information related to the non-selected factors.

XFactors divides the latent space into two subspaces:
* An orthogonal subspace $T$ (target) containing the disentangled factors.
* A subspace $S$ containing information related to the image that is either unannotated or corresponds to factors that should not be disentangled.

Our method also allows for the selection of specific directions in subspace $T$ where a factor $f_i$ will be encoded. This preserves the integrity of the latent vector's information by modifying only the intended directions to alter a single factor.

We also ensure that each factor $f_i$ is effectively disentangled across all scales and that subspace $S$ contains no information regarding the factors present in subspace $T$.

**NB: An image "factor" refers to any visual element composing the image.**

---

## Codebase organization
The repository is organized as follows:

1.  **Arch:** Contains a VAE module and a `multi_distillme` module. The `multi_distillme` module includes `train_x.py` and `xfactors.py`.
2.  **Data:** Contains `.npz` files for Cars3D, BloodMNIST, 3DShapes, DSprites, CelebA, and MNIST. All files use "label" and "image" keys.
3.  **Dataset:** Contains PyTorch `Dataset` classes for each dataset. Data is converted to `torch.float32` and scaled to $[0, 1]$.
4.  **Data_module:** Contains `DataModule` classes for each dataset. Handles automatic generation of missing `.npz` files and includes methods for loading `DataLoaders` (e.g., `Cars3DDataModule.train_dataloader()`).
5.  **Logs:** Contains training logs.
6.  **Loss:** Contains `loss.py`.
7.  **Metric:** Contains code for the FactorVAE score and DCI.
8.  **Notebook:** Primarily contains `xfactors.ipynb` for training via notebook (logs are saved in `../lightning_logs`) and `metric.ipynb` for calculating metrics from `.ckpt` files (found in the logs).
9.  **Scripts:** For command-line training using `sweep_x.sh`.
10. **Utils:** Contains constants and visualization functions.

---

## How to use sweep_x.sh
The `sweep_x.sh` script iteratively calls `train_x.sh`. Model hyperparameters are configurable within `train_x.sh`, including:

1.  Dataset selection (e.g., "shapes" for 3DShapes).
2.  $\beta_t$
3.  $\dim_s$
4.  `batch_size` and number of epochs.
5.  The dimensions allocated to each factor in $T$.
6.  An optional key for the training folder name to identify encoded factors at a glance.

The `version` variable in `sweep_x.sh` determines the log directory:
* `version=x_with_beta_t1` for $\beta_t=1$
* `version=x_with_beta_t100` for $\beta_t=100$
* `version=x_with_beta_t1_dim_t3` for $\dim_t=3$

Setting `version=MyVersion` creates the directory `logs/MyVersion/`.
The `gpu` variable in `sweep_x.sh` selects the target GPU.

From the terminal (in `script/train/`):
`./sweep_x.sh \beta_{s_1} \ldots \beta_{s_k}`

This launches $k$ training sessions via **tmux** on the selected GPU. Each tmux window trains the model with a different $\beta_s$.

**NB: Arguments must be passed as floats in the terminal (e.g., 1.0, not 1).**

---

## Navigating the logs directory
The directory structure is as follows:

1.  **Root:** `logs/` contains "version" folders like `x_with_beta_t1/`.
2.  **Versions:** Contains dataset folders (`cars3d/`, `mpi3d/`, etc.).
3.  **Datasets:** Contains `factor=` folders.

The folder `factor0,1,2,3,4` for dsprites indicates factors 0 through 4 were placed in $T$.
The folder `factor_s=-1` indicates all factors are in $T$ except the last one, which is in $S$ (default).

**NB: To change tracked factors, modify the `select_factors` variable in `train_x.py`. By default, all factors except the last one are assigned to $T$.**

The `factor=` folders contain subfolders like `test_dims2/`, `test_dim126/`, etc. These contain the training logs in the format: `x_epoch=100_beta=()_latent=()_batch=...`.

Each epoch logs reconstructions, generations, and latent space visualizations for both training and validation.

---

## Calculating metrics
The simplest way to calculate metrics is via `metric.ipynb` in the FactorVAE/XFactors section.
The `FactorVAEScore` and `DCIScore` classes only require the `.ckpt` file (found in the logs) as input.
