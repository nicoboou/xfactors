###
# mpi3d 100 8 (100)
# mpi3d 100 2

import os
PROJECT_PATH = "/projects/compures/alexandre/disdiff_adapters"
os.chdir(PROJECT_PATH)
print(os.getcwd())

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from scipy.special import digamma, gammaln
import numpy as np
from collections import Counter, defaultdict
from math import log
import seaborn as sns
from glob import glob

from torch.utils.data import DataLoader, TensorDataset
from lightning import Trainer


import torch
from sklearn.decomposition import PCA
from os.path import join
from os import mkdir
from collections import Counter, defaultdict
from tqdm import tqdm
import yaml
import re

from disdiff_adapters.data_module import LatentDataModule

from tqdm import tqdm

#DataModule
from disdiff_adapters.data_module import *
#Dataset
from disdiff_adapters.dataset import *
#Module
from disdiff_adapters.arch.multi_distillme import *
#utils
from disdiff_adapters.utils import *
#loss   
from disdiff_adapters.loss import *
#metric
from disdiff_adapters.metric import FactorVAEScore

from disdiff_adapters.arch.multi_distillme.xfactors import Xfactors
BATCH_SIZE = 2**19
LATENT_DIM_S = 126
LATENT_DIM_T = 2
is_pca = False
pref_gpu = 0
n_iter = 10000
import json
from pathlib import Path

torch.set_float32_matmul_precision('medium')

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class FactorVAEScore:

    def __init__(self, ckpt_path: str,
                is_pca: bool=False,
                n_iter: int=153600,
                batch_size: int=64,
                pref_gpu: int=0,
                verbose: bool=False,
                only_factors: list[int]=[],
                collapse_others_to_s: bool=True):

        self.ckpt_path = ckpt_path
        self.data_name = self.get_data_name()
        self.is_pca = is_pca
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.pref_gpu = pref_gpu

        # -------------------------
        # NEW: init targets AVANT load_latent()
        # -------------------------
        self.only_factors = list(only_factors) if only_factors else []
        self.collapse_others_to_s = collapse_others_to_s

        # Dédoublonne en gardant l'ordre (pas besoin que ce soit trié)
        self._target_factors = list(dict.fromkeys(self.only_factors))
        # Pour que les "if self.only_factors" restent cohérents partout
        self.only_factors = self._target_factors

        # Load latents (load_latent peut maintenant utiliser _target_factors sans crash)
        z_s_te, z_t_te, label_te = self.load_latent(stage="test", verbose=verbose)
        z_s_val, z_t_val, label_val = self.load_latent(stage="val", verbose=verbose)

        Z_te = torch.cat([z_s_te, z_t_te], dim=1).cpu().numpy()
        Z_te = (Z_te - Z_te.mean(axis=0, keepdims=True)) / (Z_te.std(axis=0, keepdims=True) + 1e-8)
        Y_te = label_te.cpu().numpy().astype(np.int64)

        Z_val = torch.cat([z_s_val, z_t_val], dim=1).cpu().numpy()
        Z_val = (Z_val - Z_val.mean(axis=0, keepdims=True)) / (Z_val.std(axis=0, keepdims=True) + 1e-8)
        Y_val = label_val.cpu().numpy().astype(np.int64)

        # -------------------------
        # NEW: configuration "targets + s"
        # -------------------------
        self.n_factors_full = Y_val.shape[1]

        if self.only_factors:
            assert max(self.only_factors) < self.n_factors_full, "Index de facteur hors-borne."

            if self.collapse_others_to_s:
                target_set = set(self._target_factors)
                self._other_factors = [i for i in range(self.n_factors_full) if i not in target_set]

                if len(self._other_factors) == 0:
                    raise ValueError(
                        "collapse_others_to_s=True mais aucun 'other factor' (tu as sélectionné tous les facteurs)."
                    )

                self._s_label = len(self._target_factors)       # ex: 6
                self._n_eval_factors = self._s_label + 1         # ex: 7
            else:
                self._other_factors = []
                self._s_label = None
                self._n_eval_factors = len(self._target_factors)
        else:
            self._other_factors = None
            self._s_label = None
            self._n_eval_factors = self.n_factors_full

        # Backward compat: ancien mode "slice" (si collapse_others_to_s=False)
        if self.only_factors and (not self.collapse_others_to_s):
            Y_val = Y_val[:, self.only_factors]
            Y_te  = Y_te[:, self.only_factors]

        self.mus_test = Z_te.T
        self.ys_test = Y_te.T
        if verbose: print("Test data formated.")

        self.mus_val = Z_val.T
        self.ys_val = Y_val.T
        if verbose: print("Val data formated.")

        self.rng = np.random.default_rng(0)

    def get_data_name(self):
        root_path = self.ckpt_path.split("/")[:-2]
        data_name = str(root_path[-6])
        return data_name

    def load_latent(self, stage: str="test", verbose: bool=False):
        latent = LatentDataModule(
            standard=True,
            batch_size=2**19,
            pref_gpu=self.pref_gpu,
            Model_class=Xfactors,
            data_name=self.data_name,
            ckpt_path=self.ckpt_path,
            verbose=verbose
        )

        latent.prepare_data()
        latent.setup(stage)
        latent_loader = latent.test_dataloader() if stage == "test" else latent.val_dataloader()
        batch = next(iter(latent_loader))
        z_s, z_t, label = batch

        # Full names (40)
        self.FACTOR_NAMES_FULL = latent.Data_class.Params.FACTORS_IN_ORDER

        # Names used by the metric output
        if self.only_factors and self.collapse_others_to_s:
            target_names = [self.FACTOR_NAMES_FULL[i] for i in self._target_factors]
            self.FACTOR_NAMES = target_names + ["s"]  # length = K+1
            self._label_to_name = {i: self.FACTOR_NAMES[i] for i in range(len(self.FACTOR_NAMES))}
        elif self.only_factors:
            self.FACTOR_NAMES = list(np.asarray(self.FACTOR_NAMES_FULL, dtype=str)[self.only_factors])
            self._label_to_name = {i: self.FACTOR_NAMES[i] for i in range(len(self.FACTOR_NAMES))}
        else:
            self.FACTOR_NAMES = self.FACTOR_NAMES_FULL
            self._label_to_name = {i: self.FACTOR_NAMES[i] for i in range(len(self.FACTOR_NAMES))}

        if self.is_pca:
            pca_t = PCA(n_components=1)
            pca_s = PCA(n_components=1)
            z_t = pca_t.fit_transform(z_t)
            z_s = pca_s.fit_transform(z_s)
            if not isinstance(z_t, torch.Tensor): z_t = torch.tensor(z_t)
            if not isinstance(z_s, torch.Tensor): z_s = torch.tensor(z_s)

        return z_s, z_t, label

    def value_index(self, ys):
        out = []
        for k in range(ys.shape[0]):
            d = {}
            for v in np.unique(ys[k]):
                d[int(v)] = np.flatnonzero(ys[k] == v)
            out.append(d)
        return out

    def collect(self, mus, ys, n_iter, batch_size=64, verbose=False):
        z_std = mus.std(axis=1, keepdims=True)
        z_std[z_std == 0] = 1.0

        v2i = self.value_index(ys)
        argmins, labels = [], []

        if verbose: print("Starting computing FactorVAE metric.")

        for _ in tqdm(range(n_iter)):
            if self.only_factors and self.collapse_others_to_s:
                # k_eval in {0..K} where K is "s"
                k_eval = self.rng.integers(0, self._n_eval_factors)

                if k_eval == self._s_label:
                    # "s" -> choose one non-target factor
                    k_src = int(self.rng.choice(self._other_factors))
                    label_k = self._s_label
                else:
                    # target k_eval -> map to the original factor index
                    k_src = int(self._target_factors[k_eval])
                    label_k = int(k_eval)

                v = self.rng.choice(list(v2i[k_src].keys()))
                pool = v2i[k_src][v]
                idx = self.rng.choice(pool, size=batch_size, replace=(len(pool) < batch_size))

                Z = mus[:, idx] / z_std
                d = int(Z.var(axis=1).argmin())
                argmins.append(d)
                labels.append(label_k)

            else:
                # original behaviour
                k = self.rng.integers(0, ys.shape[0])
                v = self.rng.choice(list(v2i[k].keys()))
                pool = v2i[k][v]
                idx = self.rng.choice(pool, size=batch_size, replace=(len(pool) < batch_size))

                Z = mus[:, idx] / z_std
                d = int(Z.var(axis=1).argmin())
                argmins.append(d)
                labels.append(k)

        return np.array(argmins), np.array(labels)

    def save(self):
        paths = self.ckpt_path.split("/")
        folder_path = ""
        for k in range(len(paths) - 2):
            folder_path += paths[k] + "/"
        folder_path += "metric"
        print(f"Saving at {folder_path}.")

        try:
            mkdir(folder_path)
        except FileExistsError:
            pass

        scores = {"dim_factor_score": self.dim_factor_score,
                "factor_dim_score": self.factor_dim_score}
        torch.save(scores, join(folder_path, "metric.pt"))

    def get_dicts(self, mu_s, ys, verbose: bool=True):
        argmins, labels = self.collect(mu_s, ys, self.n_iter, self.batch_size, verbose=verbose)
        self.argmins = argmins
        self.labels = labels

        dim_factor_score = defaultdict(list)
        for d in np.unique(argmins):
            dim_factor_score[str(d)] = defaultdict(float)
            cnt = Counter(labels[argmins == d])
            total = sum(cnt.values())
            if verbose: print(f"\nDimension {d}:")
            for k, n in cnt.most_common():
                name = self._label_to_name[int(k)]
                dim_factor_score[str(d)][name] = n / total
                if verbose: print(f"  {name:12s} : {n/total:5.1%}  ({n}/{total})")

        factor_dim_score = defaultdict(list)
        for k in np.unique(labels):
            mask = (labels == k)
            cnt = Counter(argmins[mask])
            total = sum(cnt.values())
            name = self._label_to_name[int(k)]
            factor_dim_score[name] = defaultdict(float)
            if verbose: print(f"\nFacteur {name}:")
            for d, n in cnt.most_common():
                if verbose: print(f"  dim {d:>3} : {n/total:5.1%}  ({n}/{total})")
                factor_dim_score[name][str(d)] = n / total

        self.factor_dim_score = factor_dim_score
        self.dim_factor_score = dim_factor_score

        self.save()
        return factor_dim_score, dim_factor_score

    def compute_map(self):
        dim_factor_score = self.dim_factor_score
        map_dim_factor = {}

        # plus robuste que "last key"
        if len(dim_factor_score) == 0:
            self.map_dim_factor = {}
            return self.map_dim_factor

        max_dim = max(int(k) for k in dim_factor_score.keys() if k.isdigit())
        for dim in range(max_dim + 1):
            factors = dim_factor_score[str(dim)]
            # Si aucune dim n'a été assignée -> default list
            if isinstance(factors, list) or len(factors) == 0:
                first_factor = "s"
            else:
                first_factor = list(factors.keys())[0]
            map_dim_factor[str(dim)] = first_factor

        self.map_dim_factor = map_dim_factor
        return map_dim_factor

    def compute_score(self, verbose=False):
        self.get_dicts(self.mus_val, self.ys_val, verbose=verbose)
        self.compute_map()

        argmins, labels = self.collect(self.mus_test, self.ys_test, self.n_iter, self.batch_size)

        predictions = []
        for argmin in argmins:
            pred_str = self.map_dim_factor[str(argmin)]

            if pred_str == "s":
                if self.only_factors and self.collapse_others_to_s:
                    pred_int = self._s_label
                else:
                    pred_int = -1
            else:
                try:
                    pred_int = self.FACTOR_NAMES.index(pred_str)
                except ValueError:
                    pred_int = -1

            predictions.append(pred_int)

        predictions = np.asarray(predictions)
        return np.sum(predictions == labels) / self.n_iter


class DCIscore:

    def __init__(self,
                ckpt_path: str,
                is_pca: bool = False,
                n_samples_tr=None,
                n_samples_te=None,
                only_factors: list[int] = [],
                collapse_others_to_s: bool = True,
                pref_gpu: int = 0,
                seed: int = 0):

        self.ckpt_path = ckpt_path
        self.data_name = self.get_data_name()
        self.is_pca = is_pca
        self.pref_gpu = pref_gpu

        self.only_factors = list(only_factors) if only_factors else []
        self.collapse_others_to_s = collapse_others_to_s

        # dedupe en gardant l'ordre
        self._target_factors = list(dict.fromkeys(self.only_factors))
        self.only_factors = self._target_factors

        self.rng = np.random.default_rng(seed)

        z_s_tr, z_t_tr, label_tr, z_s_te, z_t_te, label_te = self.load_latent()

        z_tr = torch.cat([z_s_tr, z_t_tr], dim=1).cpu().numpy()
        self.z_tr = (z_tr - z_tr.mean(axis=0, keepdims=True)) / (z_tr.std(axis=0, keepdims=True) + 1e-8)
        self.y_tr_full = label_tr.cpu().numpy().astype(np.int64)   # (N, K_full)

        z_te = torch.cat([z_s_te, z_t_te], dim=1).cpu().numpy()
        self.z_te = (z_te - z_te.mean(axis=0, keepdims=True)) / (z_te.std(axis=0, keepdims=True) + 1e-8)
        self.y_te_full = label_te.cpu().numpy().astype(np.int64)   # (N, K_full)

        # Samples
        self.n_samples_tr = self.y_tr_full.shape[0] if n_samples_tr is None else int(n_samples_tr)
        self.n_samples_te = self.y_te_full.shape[0] if n_samples_te is None else int(n_samples_te)

        # --- config facteurs évalués ---
        self.n_factors_full = self.y_tr_full.shape[1]

        if self.only_factors:
            assert max(self.only_factors) < self.n_factors_full, "Index de facteur hors-borne."

            if self.collapse_others_to_s:
                target_set = set(self._target_factors)
                self._other_factors = [i for i in range(self.n_factors_full) if i not in target_set]
                if len(self._other_factors) == 0:
                    raise ValueError("collapse_others_to_s=True mais aucun autre facteur (tu as tout sélectionné).")

                self._s_label = len(self._target_factors)          # ex: 6
                self._n_eval_factors = self._s_label + 1            # ex: 7

                # noms: 6 cibles + s
                target_names = [self.FACTOR_NAMES_FULL[i] for i in self._target_factors]
                self.FACTOR_NAMES = target_names + ["s"]
            else:
                self._other_factors = []
                self._s_label = None
                self._n_eval_factors = len(self._target_factors)

                self.FACTOR_NAMES = [self.FACTOR_NAMES_FULL[i] for i in self._target_factors]
        else:
            self._other_factors = None
            self._s_label = None
            self._n_eval_factors = self.n_factors_full
            self.FACTOR_NAMES = self.FACTOR_NAMES_FULL

    def load_latent(self):
        latent = LatentDataModule(
            standard=True,
            batch_size=2**19,
            Model_class=Xfactors,
            pref_gpu=self.pref_gpu,
            data_name=self.data_name,
            ckpt_path=self.ckpt_path
        )

        latent.prepare_data()

        latent.setup("val")
        batch = next(iter(latent.val_dataloader()))
        z_s_tr, z_t_tr, label_tr = batch

        latent.setup("test")
        batch = next(iter(latent.test_dataloader()))
        z_s_te, z_t_te, label_te = batch

        self.FACTOR_NAMES_FULL = latent.Data_class.Params.FACTORS_IN_ORDER

        if self.is_pca:
            # train fit
            pca_t = PCA(n_components=1)
            pca_s = PCA(n_components=1)
            z_t_tr_np = pca_t.fit_transform(z_t_tr)
            z_s_tr_np = pca_s.fit_transform(z_s_tr)

            # test transform (PAS fit_transform)
            z_t_te_np = pca_t.transform(z_t_te)
            z_s_te_np = pca_s.transform(z_s_te)

            z_t_tr = torch.tensor(z_t_tr_np)
            z_s_tr = torch.tensor(z_s_tr_np)
            z_t_te = torch.tensor(z_t_te_np)
            z_s_te = torch.tensor(z_s_te_np)

        return z_s_tr, z_t_tr, label_tr, z_s_te, z_t_te, label_te

    def get_data_name(self):
        root_path = self.ckpt_path.split("/")[:-2]
        return str(root_path[-6])

    def train_reg(self):
        """
        En mode collapse_others_to_s=True :
        - k_eval=0..K-1 : régression 1D sur facteur cible
        - k_eval=K      : régression multi-output sur tous les autres facteurs (le bucket s)
        """
        regressors = {str(k): {} for k in range(self._n_eval_factors)}

        for k_eval in tqdm(range(self._n_eval_factors)):
            reg = RandomForestRegressor(
                n_estimators=20,
                max_depth=20,
                n_jobs=-1
            )

            # --------- définir y_tr / y_te pour ce "facteur évalué" ----------
            if self.only_factors and self.collapse_others_to_s:
                if k_eval == self._s_label:
                    # s = tous les autres facteurs (multi-output)
                    y_tr = self.y_tr_full[:, self._other_factors]  # (N, K_other)
                    y_te = self.y_te_full[:, self._other_factors]
                else:
                    k_src = self._target_factors[k_eval]
                    y_tr = self.y_tr_full[:, k_src]               # (N,)
                    y_te = self.y_te_full[:, k_src]
            elif self.only_factors:
                # ancien mode : uniquement les facteurs sélectionnés, pas de "s"
                k_src = self._target_factors[k_eval]
                y_tr = self.y_tr_full[:, k_src]
                y_te = self.y_te_full[:, k_src]
            else:
                # full factors
                y_tr = self.y_tr_full[:, k_eval]
                y_te = self.y_te_full[:, k_eval]

            # --------- subsample ----------
            perm_tr = self.rng.permutation(len(self.z_tr))[:self.n_samples_tr]
            perm_te = self.rng.permutation(len(self.z_te))[:self.n_samples_te]

            reg.fit(self.z_tr[perm_tr], y_tr[perm_tr])

            score_tr = reg.score(self.z_tr[perm_tr], y_tr[perm_tr])
            score_te = reg.score(self.z_te[perm_te], y_te[perm_te])

            name = self.FACTOR_NAMES[k_eval] if k_eval < len(self.FACTOR_NAMES) else str(k_eval)
            print(f"Reg[{k_eval}] ({name}) score_tr={score_tr:.4f}, score_te={score_te:.4f}")

            regressors[str(k_eval)]["model"] = reg
            regressors[str(k_eval)]["score_tr"] = float(score_tr)
            regressors[str(k_eval)]["score_te"] = float(score_te)

        self.regressors = regressors

    def compute_weights(self):
        # D dims latentes, K facteurs évalués (incluant s si activé)
        D = self.z_tr.shape[1]
        K = self._n_eval_factors

        R = np.zeros((D, K), dtype=float)
        for k in range(K):
            model = self.regressors[str(k)]["model"]
            imp = getattr(model, "feature_importances_", None)
            if imp is None:
                raise ValueError(f"Aucune feature_importances_ pour k={k}")
            if len(imp) != D:
                raise ValueError(f"Dim mismatch: len(imp)={len(imp)} vs D={D}")
            R[:, k] = imp

        col_sum = R.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        self.P_d_given_k = R / col_sum            # (D, K)

        row_sum = R.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        self.P_k_given_d = R / row_sum            # (D, K)

    def dci_scores(self, eps=1e-12):
        r2_per_factor = [reg["score_te"] for reg in self.regressors.values()]

        # self.P_d_given_k est (D, K) => (K, D)
        R = np.asarray(self.P_d_given_k.T, dtype=float)   # (K, D)
        K, D = R.shape

        R = np.clip(R, 0.0, None)
        if R.sum() == 0:
            raise ValueError("La matrice d'importances est nulle.")

        w_d = R.sum(axis=0)
        w_k = R.sum(axis=1)
        w_d = w_d / (w_d.sum() + eps)
        w_k = w_k / (w_k.sum() + eps)

        P_k_given_d = R / (R.sum(axis=0, keepdims=True) + eps)   # (K, D)
        P_d_given_k = R / (R.sum(axis=1, keepdims=True) + eps)   # (K, D)

        def entropy(p, axis):
            p = np.clip(p, eps, 1.0)
            return -(p * np.log(p)).sum(axis=axis)

        H_k_given_d = entropy(P_k_given_d, axis=0)  # (D,)
        H_d_given_k = entropy(P_d_given_k, axis=1)  # (K,)

        D_score = float(((1.0 - H_k_given_d / (np.log(K) + eps)) * w_d).sum())
        C_score = float(((1.0 - H_d_given_k / (np.log(D) + eps)) * w_k).sum())

        I_score = None
        if r2_per_factor is not None:
            r2 = np.asarray(r2_per_factor, dtype=float)
            I_score = float(np.clip(r2, 0.0, 1.0).mean())

        return D_score, C_score, I_score

    def compute(self):
        self.train_reg()
        self.compute_weights()
        D, C, I = self.dci_scores()
        print(f"D={D}")
        print(f"C={C}")
        print(f"I={I}")
        return D, C, I


metrics_x = load_json(Path("metrics_complete.json"))
ckpt_path_x = load_json(Path("ckpt_path_x.json"))
datamodules = {"cars3d": Cars3DDataModule, "mpi3d": MPI3DDataModule, "shapes": Shapes3DDataModule, "dsprites": DSpritesDataModule,
                "celeba": CelebADataModule}

def _to_bchw(images: torch.Tensor) -> torch.Tensor:
    """Accept CHW, HWC, BCHW, BHWC -> return BCHW."""
    if images.ndim == 3:
        # CHW
        if images.shape[0] in (1, 3):
            return images.unsqueeze(0)
        # HWC
        if images.shape[-1] in (1, 3):
            return images.permute(2, 0, 1).unsqueeze(0)
        raise ValueError(f"Ambiguous image shape {tuple(images.shape)} (not CHW/HWC).")

    if images.ndim == 4:
        # BCHW
        if images.shape[1] in (1, 3):
            return images
        # BHWC
        if images.shape[-1] in (1, 3):
            return images.permute(0, 3, 1, 2)
        raise ValueError(f"Ambiguous batch shape {tuple(images.shape)} (not BCHW/BHWC).")

    raise ValueError(f"Expected 3D or 4D images, got {images.ndim}D with shape {tuple(images.shape)}")

def _minmax01_per_sample(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Min-max scale each sample independently to [0, 1] over (C,H,W).
    Expects NCHW float tensor.
    """
    x = x.float()
    mn = x.amin(dim=(1, 2, 3), keepdim=True)
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    denom = (mx - mn).clamp_min(eps)
    return (x - mn) / denom



for dim in metrics_x.keys():
    if dim == "126" : continue
    for dataset in metrics_x[dim].keys():
        if dataset == "celeba" : 
            only_factors = CelebA.Params.REPRESENTANT_IDX
            collapse_others_to_s=True
        else : 
            only_factors = []
            collapse_others_to_s=False

        #prepare for reco
        datamodule = datamodules[dataset](batch_size=512)
        datamodule.prepare_data()
        datamodule.setup("test")
        images, _ = next(iter(datamodule.test_dataloader()))
        images = torch.as_tensor(images)
        images = _to_bchw(images)
        images = _minmax01_per_sample(images)

        for bt in metrics_x[dim][dataset].keys():
            for bs in metrics_x[dim][dataset][bt].keys():
                for config in metrics_x[dim][dataset][bt][bs].keys():
                    path = ckpt_path_x[dim][dataset][bt][bs][config]
                    if path != "" and config == "n-1":
                        try :
                            #reco
                            xfactors = Xfactors.load_from_checkpoint(path, map_location=set_device(pref_gpu)[0])
                            images = images.to(device=xfactors.device, dtype=torch.float32)
                            recos = []
                            for _ in range(5) : recos.append(mse(xfactors(images)[2], images).item())
                            recos = np.asarray(recos)
                            metrics_x[dim][dataset][bt][bs][config]["reco"] = (float(recos.mean()), float(recos.std()))

                            # if type(metrics_x[dim][dataset][bt][bs][config]["score"]) != list : 
                            #     #FactorVAEScore
                            #     score = FactorVAEScore(path, only_factors=only_factors, collapse_others_to_s=collapse_others_to_s,
                            #                 n_iter=n_iter, pref_gpu=pref_gpu)
                            #     scores = []
                            #     for _ in range(5): scores.append(score.compute_score())
                            #     scores = np.asarray(scores)
                            #     metrics_x[dim][dataset][bt][bs][config]["score"] = (float(scores.mean()), float(scores.std()))

                            #     #DCI
                            #     dci = DCIscore(path, only_factors=only_factors, collapse_others_to_s=collapse_others_to_s, pref_gpu=pref_gpu)
                            #     dcis = {"d":[], "c":[], "i":[]}
                            #     for _ in range(5) : 
                            #         d,c,i = dci.compute()
                            #         dcis["d"].append(d)
                            #         dcis["c"].append(c)
                            #         dcis["i"].append(i)
                            #     dcis["d"] = np.asarray(dcis["d"])
                            #     dcis["c"] = np.asarray(dcis["c"])
                            #     dcis["i"] = np.asarray(dcis["i"])

                            #     metrics_x[dim][dataset][bt][bs][config]["d"] = (float(dcis["d"].mean()), float(dcis["d"].std()))
                            #     metrics_x[dim][dataset][bt][bs][config]["c"] = (float(dcis["c"].mean()), float(dcis["c"].std()))
                            #     metrics_x[dim][dataset][bt][bs][config]["i"] = (float(dcis["i"].mean()), float(dcis["i"].std()))

                        except Exception as e : 
                                print(e)
                                metrics_x[dim][dataset][bt][bs][config]["d"] = -2
                                metrics_x[dim][dataset][bt][bs][config]["c"] = -2
                                metrics_x[dim][dataset][bt][bs][config]["i"] = -2
                                metrics_x[dim][dataset][bt][bs][config]["score"] = -2
                                metrics_x[dim][dataset][bt][bs][config]["reco"] = -2

with open("metrics_test.json", "w") as f:
    json.dump(metrics_x, f, indent=2, ensure_ascii=False)