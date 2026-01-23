import torch
import numpy as np
from sklearn.decomposition import PCA
from os.path import join
from os import mkdir
from collections import Counter, defaultdict
from tqdm import tqdm
import yaml
import re

from disdiff_adapters.data_module import LatentDataModule

class FactorVAEScore :

    def __init__(self, ckpt_path: str, is_pca: bool=False, n_iter=153600, batch_size=64) :
        self.ckpt_path = ckpt_path
        self.data_name = self.get_cfg()[-1]
        self.is_pca = is_pca
        self.n_iter = n_iter
        self.batch_size = batch_size

        z_s_te, z_t_te, label_te = self.load_latent()                                  

        Z_te = torch.cat([z_s_te, z_t_te], dim=1).cpu().numpy()        
        Z_te = (Z_te - Z_te.mean(axis=0, keepdims=True)) / (Z_te.std(axis=0, keepdims=True) + 1e-8)
        Y_te = label_te.cpu().numpy().astype(np.int64)              

        self.mus_test = Z_te.T                                   
        self.ys_test  = Y_te.T   
        print("Test data formated.") 

        self.rng = np.random.default_rng(0)
    
    def set_batch_size(self, batch_size: int) :
        self.batch_size = batch_size

    def set_n_iter(self, n_iter: int):
        self.n_iter = n_iter
        
    def load_latent(self):
        latent = LatentDataModule(standard=True, 
                                batch_size=2**19,
                                pref_gpu=2,
                                data_name=self.data_name,
                                ckpt_path=self.ckpt_path)
        
        print("Prepare data: test if .npz files exist.")
        latent.prepare_data()

        print("Start loading test batch.")
        latent.setup("test")
        print("Start loading test batch. - end setup")
        latent_test_loader = latent.test_dataloader()
        print("Start loading test batch. -end dataloader")
        batch = next(iter(latent_test_loader))
        z_s_te, z_t_te, label_te = batch
        print(f"Test batch shape: {z_s_te.shape, z_t_te.shape, label_te.shape}")

        self.FACTOR_NAMES = latent.Data_class.Params.FACTORS_IN_ORDER

        if self.is_pca:
            print("Start PCA.")
            #test
            pca_t_te = PCA(n_components=1) 
            pca_s_te = PCA(n_components=1)
            z_t_te = pca_t_te.fit_transform(z_t_te)
            z_s_te = pca_s_te.fit_transform(z_s_te)
            print(z_s_te.shape, z_t_te.shape)
            if not isinstance(z_t_te, torch.Tensor) : z_t_te = torch.tensor(z_t_te)
            if not isinstance(z_s_te, torch.Tensor) : z_s_te = torch.tensor(z_s_te)
            print("End PCA.")
        
        return z_s_te, z_t_te, label_te

    def value_index(self, ys):
        out=[]
        for k in range(ys.shape[0]):
            d={}
            for v in np.unique(ys[k]):
                d[int(v)]=np.flatnonzero(ys[k]==v)
            out.append(d)
        return out

    def collect(self, mus, ys, n_iter, batch_size=64):
        z_std = mus.std(axis=1, keepdims=True); z_std[z_std==0]=1.0
        v2i = self.value_index(ys)
        argmins, labels = [], []
        print("Starting computing FactorVAE metric.")
        for _ in tqdm(range(n_iter)):
            k = self.rng.integers(0, ys.shape[0]) #Choose a factor f_k
            v = self.rng.choice(list(v2i[k].keys())) #Choose a value for f_k
            pool = v2i[k][v]
            idx = self.rng.choice(pool, size=batch_size, replace=(len(pool)<batch_size)) #Batch with f_k=v

            Z = mus[:, idx]/z_std
            d = int(Z.var(axis=1).argmin()) #get the argmin variance for this batch
            argmins.append(d); labels.append(k)
        return np.array(argmins), np.array(labels)
    
    def save(self):
        paths = self.ckpt_path.split("/")
        folder_path=""
        for k in range(len(paths)-2) :
            folder_path+= paths[k]+"/"
        folder_path+="metric"
        print(f"Saving at {folder_path}.")

        try: mkdir(folder_path)
        except FileExistsError: pass

        scores = {"dim_factor_score": self.dim_factor_score, "factor_dim_score": self.factor_dim_score}
        torch.save(scores, join(folder_path, "metric.pt"))
        
    def get_argmins(self, verbose: bool=True) :
        argmins, labels = self.collect(self.mus_test, self.ys_test, self.n_iter, self.batch_size)
        self.argmins = argmins
        self.labels = labels
        dim_factor_score = {}
        # Taux d'association dim->facteur
        for d in np.unique(argmins):
            dim_factor_score[str(d)] = defaultdict(float)
            cnt = Counter(labels[argmins==d]) #labels[argmins==d], How many times f_k is assigned to dimension d?
            total = sum(cnt.values()) #Number of labels assigned to dimension d
            if verbose: print(f"\nDimension {d}:")
            for k,n in cnt.most_common():
                dim_factor_score[str(d)][self.FACTOR_NAMES[k]] = n/total
                if verbose: print(f"  {self.FACTOR_NAMES[k]:12s} : {n/total:5.1%}  ({n}/{total})")

        # """Dimension 0:
        #   scale        : 73.8%  (135/183)
        #   shape        : 26.2%  (48/183)
        # """ means 73.8% of labels assigned to dimension 0 are scale.


        factor_dim_score = {}
        for k in np.unique(labels):
            mask = (labels == k)
            cnt = Counter(argmins[mask])                # Combien de fois la dim d "gagne" pour le facteur k ?
            total = sum(cnt.values())
            name  = self.FACTOR_NAMES[k]
            factor_dim_score[name] = defaultdict(float)
            if verbose: print(f"\nFacteur {name}:")
            for d, n in cnt.most_common():             # tri décroissant
                if verbose: print(f"  dim {d:>3} : {n/total:5.1%}  ({n}/{total})")
                factor_dim_score[name][str(d)]=n/total

        self.factor_dim_score = factor_dim_score
        self.dim_factor_score = dim_factor_score

        self.save()

    def safe_load_yaml(self, path):
        try:
            with open(path, "r") as f:
                text = f.read()
            # Supprimer les lignes contenant !!python/name:
            text = re.sub(r"!!python/name:[^\n]*", "''", text)
            return yaml.safe_load(text)
        except FileNotFoundError as e : raise e

    def get_cfg(self):
        root_path = self.ckpt_path.split("/")[:-2]
        data_name = str(root_path[-6])
        root_path.append("hparams.yaml")
        hparams_path = join(*root_path)
        hparams_path="/"+hparams_path
        try : 
            cfg = self.safe_load_yaml(hparams_path)
            return int(cfg["latent_dim_t"]), int(cfg["latent_dim_s"]), int(cfg["select_factor"]), str(data_name)
        except FileNotFoundError as e : 
            print(e)
            return 0, 0, 0

    def get_score(self):

        self.get_argmins()
        N = len(self.argmins)
        tp = 0
        dim_t, dim_s, select_factor, _ = self.get_cfg()
        dims_t = [dim_s+k for k in range(dim_t)]

        for dim, factor in zip(self.argmins, self.labels):

            if dim in dims_t :
                if factor == select_factor : tp+=1
            if dim not in dims_t :
                if factor != select_factor : tp+=1
        score = tp/N
        print(f"FactorVAEScore: {score}")
        return score

