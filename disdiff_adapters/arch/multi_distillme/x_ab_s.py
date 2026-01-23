import torch
import torch.nn as nn
from lightning import LightningModule
import matplotlib.pyplot as plt
import os
import math
from os.path import join
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from json import dump
from tqdm import tqdm

from disdiff_adapters.arch.vae import *
from disdiff_adapters.utils import *
from disdiff_adapters.loss import *


class _MultiDistillMe(torch.nn.Module) : 
    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 latent_dim_t: int,
                 res_block: nn.Module=ResidualBlock) :
        
        super().__init__()
        
        self.encoder_t = Encoder(in_channels=in_channels, 
                                 img_size=img_size,
                                 latent_dim=latent_dim_t,
                                 res_block=res_block)

        self.decoder = Decoder(out_channels=in_channels,
                               img_size=img_size,
                               latent_dim=latent_dim_t,
                               res_block=res_block,
                               out_encoder_shape=self.encoder_t.out_encoder_shape)

    def forward(self, images: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:

        #forward_t - semble encoder la forme
        mus_logvars_t = self.encoder_t(images)
        z_t = sample_from(mus_logvars_t)

        #decoder
        image_hat_logits = self.decoder(z_t)

        return mus_logvars_t, image_hat_logits, z_t
    
class Xfactors(LightningModule) :
### TO DO
## Permettre de séléctionner des factor_value pour plusieurs facteurs
## Adapter ça dans generate_cond


    def __init__(self,
                 in_channels: int,
                 img_size: int,
                 select_factors: list[int]=[0],
                 dims_by_factors: list[int]=[2],
                 res_block: nn.Module=ResidualBlock,
                 beta_t: float=1.0,
                 warm_up: bool=False,
                 kl_weight: float= 1e-6,
                 type: str="all",
                 l_cov: float=0.0,
                 l_nce_by_factors: list[float]=[1e-3],
                 l_anti_nce: float=0.0,
                 temp: float=0.07,
                 factor_value=-1,
                 map_idx_labels: list|None= None) :
        
        super().__init__()
        assert len(l_nce_by_factors) == len(dims_by_factors) and len(dims_by_factors) == len(select_factors)
        
        self.save_hyperparameters(ignore=["res_block"])

        self.model = _MultiDistillMe(in_channels=self.hparams.in_channels,
                                     img_size=self.hparams.img_size,
                                     latent_dim_t=sum(dims_by_factors),
                                     res_block=res_block)
            

        self.images_test_buff = None
        self.images_train_buff = []
        self.labels_train_buff = []
        self.latent_train_buff:dict[str, torch.Tensor] = {"t": []}

        self.images_val_buff = []
        self.labels_val_buff = []
        self.latent_val_buff:dict[str, torch.Tensor] = {"t": []}

        self.constrastive = InfoNCESupervised(temperature=self.hparams.temp)
        self.current_batch = 0
        self.current_val_batch = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())
    
    def forward(self, images: torch.Tensor, test=False) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mus_logvars_t, image_hat_logits,  z_t = self.model(images)

        return mus_logvars_t, image_hat_logits, z_t
    
    def loss(self, 
             mus_logvars_t: torch.Tensor, 
             image_hat_logits: torch.Tensor, 
             images: torch.Tensor, 
             z_t: torch.Tensor,
             labels=None, 
             log_components: bool=False) :

        weighted_kl_t = self.hparams.kl_weight*self.hparams.beta_t*kl(*mus_logvars_t)
        reco = mse(image_hat_logits, images)
        nces = []

        num_factors = len(self.hparams.select_factors)
        start_dim = 0
        for i, d in enumerate(self.hparams.dims_by_factors) :
            end_dim = start_dim+d
            label = labels[:, self.hparams.select_factors[i]]
            nce = self.hparams.l_nce_by_factors[i]*self.constrastive(z_t[:, start_dim:end_dim], label)
            nces.append(nce)

            start_dim = end_dim

            if log_components : self.log(f"loss/nce_{self.hparams.select_factors[i]}", nce.detach())
        assert start_dim == z_t.size(1), "dims_by_factors ne couvre pas toutes les dims de z_t"

        nce = torch.stack(nces).sum()
        if log_components :
            self.log("loss/kl_t", weighted_kl_t.detach())
            self.log("loss/reco", reco.detach())


        if self.hparams.type == "all" : 
            if self.hparams.warm_up and self.current_epoch <= int(0.1*self.trainer.max_epochs) :
                loss_value = weighted_kl_t+reco+nce
            else : loss_value = weighted_kl_t+reco+nce
        elif self.hparams.type == "vae" : loss_value = weighted_kl_t+reco
        elif self.hparams.type == "vae_nce" : loss_value = weighted_kl_t+reco+nce
        elif self.hparams.type == "vae_cov" : loss_value = weighted_kl_t+reco
        elif self.hparams.type == "reco" : loss_value = reco
        elif self.hparams.type == "kl" : loss_value =  weighted_kl_t
        elif self.hparams.type == "nce" : loss_value = nce
        else : raise ValueError("Loss type error")

        return loss_value
    
    def training_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        
        mus_logvars_t, image_hat_logits, z_t = self.forward(images)
        loss = self.loss(mus_logvars_t, image_hat_logits, images, z_t, 
                         labels=labels, log_components=True)
        
        if torch.isnan(loss):
            #raise ValueError("NaN loss")
            try : loss = self.prev_loss
            except : ValueError("Nan loss")

        self.log("loss/train", loss)


        if self.current_batch <= 300 :
            self.images_train_buff.append(images.detach().cpu())
            self.labels_train_buff.append(labels.detach().cpu())
            self.latent_train_buff["t"].append(mus_logvars_t[0].detach().cpu())

        self.current_batch += 1

        return loss

    def validation_step(self, batch: tuple[torch.Tensor]):
        images, labels = batch

        mus_logvars_t, image_hat_logits, z_t = self.forward(images, test=True)
        loss = self.loss(mus_logvars_t, image_hat_logits, images, z_t, labels=labels)
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")
        self.log("loss/val", loss)

        if self.current_val_batch <= 100:
            self.images_val_buff.append(images.detach().cpu())
            self.labels_val_buff.append(labels.detach().cpu())
            self.latent_val_buff["t"].append(z_t.detach().cpu())
        self.current_val_batch += 1

        print(f"Val loss: {loss}")
    
    def test_step(self, batch: tuple[torch.Tensor]) :
        images, labels = batch
        mus_logvars_t, image_hat_logits, z_t = self.forward(images, test=True)

        weighted_kl_t = self.hparams.beta_t*kl(*mus_logvars_t)
        reco = mse(image_hat_logits, images)

        self.log("loss/reco_test", reco, sync_dist=True)
        self.log("loss/kl_t_test", weighted_kl_t, sync_dist=True)

        if self.images_test_buff is None : self.images_test_buff = images


#### Generate & Reco

    def generate(self, nb_samples: int=16, is_val: bool=False) :
        #eps_s = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_s])).to(self.device, torch.float32)
        #eps_t = torch.randn_like(torch.zeros([nb_samples, self.hparams.latent_dim_t])).to(self.device, torch.float32)

        buff_latents = self.latent_val_buff if is_val else self.latent_train_buff
        buff_labels = self.labels_val_buff if is_val else self.labels_train_buff
        buff_imgs = self.images_val_buff if is_val else self.images_train_buff

        z_t= buff_latents["t"]

        nb_sample_latent = z_t.shape[0]
        assert nb_samples <= nb_sample_latent, "Too much points"

        idx_t = torch.randint_like(torch.zeros([nb_samples]), high=nb_sample_latent, dtype=torch.int32)
        eps_t = z_t[idx_t].to(self.device)

        x_hat_logits = self.model.decoder(eps_t)
        return x_hat_logits

    def generate_by_factors(self, cond: int, nb_samples: int=16, pos: int=0, z_t=None, is_val=False,
                            img_ref=None, factor_value=-1) :
        assert cond in self.hparams.select_factors, f"Impossible to generate with cond: {cond}. The factor is not followed."


        buff_latents = self.latent_val_buff if is_val else self.latent_train_buff
        buff_labels = self.labels_val_buff if is_val else self.labels_train_buff
        buff_imgs = self.images_val_buff if is_val else self.images_train_buff

        if z_t is None : 
            z_t= buff_latents["t"]
        else : print("interactive mode : on")

        #If a factor_value should be represented, we mask the latent space to select the correct value
        if factor_value != -1 :
            mask = (buff_labels[:, cond] == factor_value)
            z_t = z_t[mask]
        else : mask = torch.ones(z_t.size(0), dtype=bool)
        if pos>=len(mask) : pos=0
        i = self.hparams.select_factors.index(cond)
        start = sum(self.hparams.dims_by_factors[:i])
        end = start+self.hparams.dims_by_factors[i]

        nb_sample_latent = z_t.shape[0]
        assert nb_samples <= nb_sample_latent, "Too much points"

        idx_cond = torch.randint_like(torch.zeros([nb_samples]), high=nb_sample_latent, dtype=torch.int32)
        eps_cond = z_t[idx_cond, start:end].to(self.device)

        if img_ref is None :
            eps_t_left = torch.stack(nb_samples*[z_t[pos, :start]]).to(self.device)
            eps_t_right = torch.stack(nb_samples*[z_t[pos, end:]]).to(self.device)
        else : 
            with torch.no_grad() : _, _, eps_full_t = self(img_ref.to(self.device), test=True)
            eps_t_left = torch.cat(nb_samples*[eps_full_t[:, :start]]).to(self.device)
            eps_t_right = torch.cat(nb_samples*[eps_full_t[:, end:]]).to(self.device)

        eps_t = torch.cat([eps_t_left, eps_cond, eps_t_right], dim=1)

        z = eps_t

        ref_img = img_ref.squeeze(0) if img_ref is not None else buff_imgs[mask][pos]
        return self.model.decoder(z).detach().cpu(), ref_img.unsqueeze(0).detach().cpu()         
    
    def show_reconstruct(self, images: torch.Tensor) :
        images = images.to(self.device)[:8]
        with torch.no_grad() : mus_logvars_t, images_gen, z_t = self(images, test=True)

        fig, axes = plt.subplots(len(images), 2, figsize=(7, 20))

        for i in range(len(images)) :
            img = images[i]
            img_gen = images_gen[i]

            images_proc = (255*((img - img.min()) / (img.max() - img.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()
            images_gen_proc = (255*((img_gen - img_gen.min()) / (img_gen.max() - img_gen.min() + 1e-8))).to("cpu",torch.uint8).permute(1,2,0).detach().numpy()

            axes[i,0].imshow(images_proc)
            axes[i,1].imshow(images_gen_proc)

            axes[i,0].set_title("original")
            axes[i,1].set_title("reco")
        plt.tight_layout()
        plt.show()
        return mus_logvars_t, images_gen, z_t

#### Lightning Hooks

    def on_train_epoch_start(self):
        self.images_train_buff = []
        self.labels_train_buff = []
        self.latent_train_buff = {"t": []}
        self.current_batch = 0

        self.images_val_buff = []
        self.labels_val_buff = []
        self.latent_val_buff = {"t": []}

    def on_train_epoch_end(self):
        epoch = self.current_epoch

        if epoch % 1 == 0:
            try : os.mkdir(os.path.join(self.logger.log_dir, f"epoch_{epoch}"))
            except FileExistsError as e : pass

            #compute a sample of the latent space
            # for images in self.images_train_buff :
            #     with torch.no_grad() : _, _, _, z_s, z_t, z = self.forward(images.to(self.device), test=True) #images shape : [32, 3, 64, 64]

            #     self.latent_train_buff["s"].append(z_s.detach().cpu())
            #     self.latent_train_buff["t"].append(z_t.detach().cpu())
            self.latent_train_buff["t"] = torch.cat(self.latent_train_buff["t"])

            self.images_train_buff = torch.cat(self.images_train_buff)
            self.labels_train_buff = torch.cat(self.labels_train_buff)
            mus_logvars_t = self.log_reco()

            self.log_gen_images()

            ### latent space
            self.log_latent()
            # self.log_factorvae()

    def on_validation_epoch_start(self):
        self.current_val_batch = 0

    def on_validation_epoch_end(self) :
        epoch = self.current_epoch

        if epoch % 1 == 0:
            try : os.mkdir(os.path.join(self.logger.log_dir, f"epoch_{epoch}"))
            except FileExistsError as e : pass
            try : os.mkdir(os.path.join(self.logger.log_dir, f"epoch_{epoch}", "val"))
            except FileExistsError as e : pass

            #compute a sample of the latent space
            # for images in self.images_val_buff :
            #     with torch.no_grad() : _, _, _, z_s, z_t, z = self.forward(images.to(self.device), test=True) #images shape : [32, 3, 64, 64]

            #     self.latent_val_buff["s"].append(z_s.detach().cpu())
            #     self.latent_val_buff["t"].append(z_t.detach().cpu())
            self.latent_val_buff["t"] = torch.cat(self.latent_val_buff["t"])

            self.images_val_buff = torch.cat(self.images_val_buff)
            self.labels_val_buff = torch.cat(self.labels_val_buff)
            mus_logvars_t = self.log_reco(is_val=True)

            self.log_gen_images(is_val=True)

            ### latent space
            self.log_latent(is_val=True)
            # self.log_factorvae(is_val=True)


#### Log

    def log_reco(self, is_val: bool=False) :
        buff_imgs = self.images_val_buff if is_val else self.images_train_buff
        mus_logvars_t, images_gen, z_t = self.show_reconstruct(buff_imgs[:8]) #display images and reconstruction in interactive mode; save the plot in plt.gcf() if non interactive
        #save the recontruction plot saved in plt.gcf()
        save_reco_path = os.path.join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"reco_{self.current_epoch}.png")
        if is_val : save_reco_path = os.path.join(self.logger.log_dir, f"epoch_{self.current_epoch}", "val" ,f"reco_{self.current_epoch}.png")

        fig = plt.gcf()
        fig.savefig(save_reco_path)
        plt.close(fig)
        return mus_logvars_t

    def log_gen_images(self, is_val: bool=False) :
        epoch = self.current_epoch

        #save the generate images
        images_gen = self.generate(is_val=is_val)
        save_gen_path = join(self.logger.log_dir, f"epoch_{epoch}", f"gen_{epoch}.png")
        if is_val : save_gen_path = join(self.logger.log_dir, f"epoch_{epoch}","val",f"gen_{epoch}.png")
        vutils.save_image(images_gen.detach().cpu(), save_gen_path)
        
        for cond in self.hparams.select_factors :
            for i in range(4) :
                factor_value = self.hparams.factor_value if i%2 else -1
                images_cond_f_gen, input_s = self.generate_by_factors(cond=cond, pos=i, factor_value=factor_value, is_val=is_val)
                images_cond_f_gen_ref = torch.cat([images_cond_f_gen.detach().cpu(), input_s])
                save_gen_f_path = join(self.logger.log_dir, f"epoch_{epoch}", f"gen_f={cond}_{epoch}_{i}.png")
                if is_val : save_gen_f_path = join(self.logger.log_dir, f"epoch_{epoch}", "val" ,f"gen_f={cond}_{epoch}_{i}.png")
                vutils.save_image(images_cond_f_gen_ref.detach().cpu(), save_gen_f_path)

        ### Merge in one image

        for cond in self.hparams.select_factors:
            if is_val : path_epoch_f = [ join(self.logger.log_dir, f"epoch_{epoch}", "val",f"gen_f={cond}_{epoch}_{i}.png") for i in range(4)]
            else : path_epoch_f = [ join(self.logger.log_dir, f"epoch_{epoch}",f"gen_f={cond}_{epoch}_{i}.png") for i in range(4)]
            final_gen_f = merge_images_with_black_gap(path_epoch_f)
            final_gen_f.save(join(self.logger.log_dir, f"final_gen_f={cond}.png"))
            for i in range(4): os.remove(path_epoch_f[i])    

        map_ = self.hparams.map_idx_labels
        label = [f"gen_{cond}" for cond in self.hparams.select_factors] if map_ is None else [f"gen_{map_[cond]}" for cond in self.hparams.select_factors]
        final_image = merge_images(save_gen_path,
                                   *[join(self.logger.log_dir, f"final_gen_f={cond}.png") for cond in self.hparams.select_factors],
                                   labels=["gen",]+label)
        if is_val : save_gen_all_path = join(self.logger.log_dir, f"epoch_{epoch}","val",f"gen_all_{epoch}.png")
        else : save_gen_all_path = join(self.logger.log_dir, f"epoch_{epoch}",f"gen_all_{epoch}.png")
        final_image.save(save_gen_all_path)


        for cond in self.hparams.select_factors: os.remove(join(self.logger.log_dir, f"final_gen_f={cond}.png"))

    def log_latent(self, is_val: bool=False) :
        buff_latents = self.latent_val_buff if is_val else self.latent_train_buff
        buff_labels = self.labels_val_buff if is_val else self.labels_train_buff
        buff_imgs = self.images_val_buff if is_val else self.images_train_buff

        number_labels = buff_labels.shape[1]
        
        for i in range(number_labels) :
            labels = buff_labels[:, i].unsqueeze(1)
            z_f_path = [join(self.logger.log_dir, f"z_f={cond}.png") for cond in self.hparams.select_factors]
            title = f"latent space s {i}" if self.hparams.map_idx_labels is None else "latent space s "+self.hparams.map_idx_labels[i]

            for j, cond in enumerate(self.hparams.select_factors) :
                title = f"latent space t_{j} {i}" if self.hparams.map_idx_labels is None else f"latent space t_{j} "+self.hparams.map_idx_labels[i]
                start = sum(self.hparams.dims_by_factors[:j])
                end = start+self.hparams.dims_by_factors[j]
                display_latent(labels=labels, z=buff_latents["t"][:, start:end], title=title)
                fig = plt.gcf()
                fig.savefig(z_f_path[j])
                plt.close(fig)
            latent_img = merge_images_with_black_gap(z_f_path)
            
            if is_val : path_latent = join(self.logger.log_dir, f"epoch_{self.current_epoch}", "val", f"latent_space_{i}_{self.current_epoch}.png")
            else : path_latent = join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"latent_space_{i}_{self.current_epoch}.png")
            latent_img.save(path_latent)
            for path in z_f_path : os.remove(path)

        if is_val : 
            path_latents = [join(self.logger.log_dir, f"epoch_{self.current_epoch}", "val", f"latent_space_{i}_{self.current_epoch}.png") for i in range(number_labels)]
            path_final_latent = join(self.logger.log_dir, f"epoch_{self.current_epoch}", "val", f"latent_space_{self.current_epoch}.png")
        else :
            path_latents = [join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"latent_space_{i}_{self.current_epoch}.png") for i in range(number_labels)]
            path_final_latent = join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"latent_space_{self.current_epoch}.png")
        
        final_images = merge_images(path_latents)
        final_images.save(path_final_latent)
        for i in range(number_labels) : os.remove(path_latents[i])


    # def log_factorvae(self, is_val:bool=False):
    #     latent_buff = self.latent_val_buff if is_val else self.latent_train_buff
    #     label = self.labels_val_buff if is_val else self.labels_train_buff 
    #     mode = "val" if is_val else "train"

    #     factorvaescore = FactorVAEScoreLight(z_s=latent_buff["s"],
    #                                          z_t=latent_buff["t"],
    #                                          label=label,
    #                                          dim_t = self.hparams.latent_dim_t,
    #                                          dim_s = self.hparams.latent_dim_s,
    #                                          select_factor=self.hparams.select_factors[0],
    #                                          n_iter=150000,
    #                                          batch_size=512)
    #     score = factorvaescore.get_score()
    #     self.log(f"{mode}/factorvae", score)

    #     save_factorvae_path = os.path.join(self.logger.log_dir, f"epoch_{self.current_epoch}", f"metric_{self.current_epoch}.json")
    #     if is_val : save_factorvae_path = os.path.join(self.logger.log_dir, f"epoch_{self.current_epoch}", "val" ,f"metric_{self.current_epoch}.json")

    #     data = {"factorvaescore": score}
    #     with open(save_factorvae_path, "w") as f: dump(data, f, indent=4)
        



class FactorVAEScoreLight :

    def __init__(self,  
                 z_s: torch.Tensor,
                 z_t: torch.Tensor,
                 label: torch.Tensor,
                 dim_t: int, 
                 dim_s: int, 
                 select_factor: int,
                 n_iter: int=100000,
                 batch_size: int=256) :

        self.format_data(z_s, z_t, label)
        self.dim_t = dim_t
        self.dim_s = dim_s
        self.select_factor = select_factor
        self.rng = np.random.default_rng(0)
        self.n_iter = n_iter
        self.batch_size = batch_size
        
    def format_data(self, z_s, z_t, label):

        Z = torch.cat([z_s, z_t], dim=1).cpu().numpy()        
        Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-8)
        Y = label.cpu().numpy().astype(np.int64)              

        self.mus = Z.T                                   
        self.ys  = Y.T
        print("data formated.")

    def value_index(self, ys):
        out=[]
        for k in range(ys.shape[0]):
            d={}
            for v in np.unique(ys[k]):
                d[int(v)]=np.flatnonzero(ys[k]==v)
            out.append(d)
        return out

    def collect(self, mus, ys, n_iter, batch_size):
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
    
    def get_argmins(self) :
        argmins, labels = self.collect(self.mus, self.ys, n_iter=self.n_iter, batch_size=self.batch_size)
        self.argmins = argmins
        self.labels = labels

    def get_score(self):
        self.get_argmins()
        N = len(self.argmins)
        tp = 0
        dims_t = [self.dim_s+k for k in range(self.dim_t)]

        for dim, factor in zip(self.argmins, self.labels):

            if dim in dims_t :
                if factor == self.select_factor : tp+=1
            if dim not in dims_t :
                if factor != self.select_factor : tp+=1
        score = tp/N
        print(f"FactorVAEScore: {score}")
        return score