import h5py
import torch
from torch import sort
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import math
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns # type: ignore
from disdiff_adapters.loss import *
import json
from pathlib import Path

from sklearn.decomposition import PCA

def load_h5(h5_path: str) :
    """
    Load datasetH5 object from a H5 file.

    Args:
        h5_path: str path of the h5 file.

    Return:
        list of datasetH5
    """
    try : 
        dataset_h5 = h5py.File(h5_path, "r")
        return [dataset_h5[key] for key in dataset_h5.keys()]
    except FileNotFoundError as e : print("WARNING : file not foud.")
    
def split(data: torch.Tensor, label: torch.Tensor, ratio: float=0.8) :
    """
    Shuffle and split (data,label) in a train, val and test set.
    The ratio is the size of the train set.
    The val and test are each a half of the remain samples.

    Args:
        data: could be an array, torch.tensor, H5dataset etc...
        label: same type

    Return:
        Tensors for each data/label set.
        train_data, train_labels, test_data, test_labels

    """
    print("start split")

    idx = torch.randperm(len(data))

    train_idx, _ = sort(idx[:int(len(data)*ratio)])
    
    test_idx, _ = sort(idx[int(len(data)*(ratio)):])

    train_data = data[train_idx]
    train_label = label[train_idx]

    test_data = data[test_idx]
    test_label = label[test_idx]
    print("end split")

    return torch.tensor(train_data), torch.tensor(train_label), torch.tensor(test_data), torch.tensor(test_label)

def display(batch: tuple[torch.Tensor]) -> None:
    """
    Affiche un batch d'images RGB.
    batch = (images [B,C,H,W], labels [B,])
    """
    images, labels = batch
    if not isinstance(images, torch.Tensor) : images = torch.from_numpy(images)
    if not isinstance(labels, torch.Tensor) : labels = torch.from_numpy(labels)
    nb_samples = images.size(0)

    # grille quasi carrée
    nb_col = math.ceil(math.sqrt(nb_samples))
    nb_row = math.ceil(nb_samples / nb_col)

    fig, axes = plt.subplots(nb_row, nb_col, figsize=(3*nb_col, 3*nb_row))

    # Toujours 2D, quelles que soient les tailles
    axes = np.atleast_1d(axes)             # -> array 1D si un seul axe
    axes = np.array(axes, dtype=object).reshape(nb_row, nb_col)

    for i in range(nb_row * nb_col):
        r, c = divmod(i, nb_col)           
        ax = axes[r, c]
        if i < nb_samples:
            img = images[i]
            # normalisation min-max par image
            img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).to(torch.uint8)
            if img.size(0) == 3 : ax.imshow(img.permute(1, 2, 0).cpu().numpy())
            else : ax.imshow(img[0].cpu().numpy(), cmap="gray")
            # titre robuste (labels scalaires ou vecteurs)
            lbl = labels[i]
            try:
                title = str(lbl.item())
            except Exception:
                title = str(lbl.detach().cpu().numpy())
            ax.set_title(title)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def build_mask(labels: torch.Tensor, select_factor: int, factor_value) -> tuple[torch.Tensor, torch.Tensor]:
    """
    labels : [N, F] ou [N]  (valeurs -1/1 ou 0/1)
    select_factor : colonne à filtrer si labels est 2D
    factor_value : valeur à garder (ex: 1, 0 ou -1)

    Retourne:
      mask    : [N] bool, True si l'échantillon est gardé
      idx_m2o : [M] long, pour un index j dans le tens. masqué, 
                          l'original vaut idx_m2o[j]
    """
    col = labels[:, select_factor] if labels.ndim == 2 else labels
    mask = (col == factor_value)                            # [N] bool
    idx_m2o = mask.nonzero(as_tuple=True)[0].to(torch.long) # [M]
    return mask, idx_m2o

def sample_from(mu_logvar: tuple[torch.Tensor], test: bool=False) -> torch.Tensor:
    mu, logvar = mu_logvar

    # clamp pour éviter exp(0.5 * logvar) -> inf
    logvar = torch.clamp(logvar, min=-20.0, max=20.0)

    if test:
        return mu

    eps = torch.randn_like(mu)
    std = torch.exp(0.5 * logvar)
    z = mu + std * eps

    # optionnel : garde-fou supplémentaire
    if not torch.isfinite(z).all():
        print("[sample_from] Non-finite in z:",
              "NaN =", torch.isnan(z).sum().item(),
              "Inf =", torch.isinf(z).sum().item())
        raise ValueError("Non-finite z in sample_from")

    return z



def del_outliers(arr: np.ndarray, k: int) -> np.ndarray:

    # assert type(arr) == np.ndarray, "should be an array"
    # assert k<=len(arr), "k>len(arr)"
    # assert len(arr.shape) == 1, "error shape"
    # idxs = np.argpartition(arr, -k)[-k:]
    # arr[idxs] = 0
    return arr


def hex_to_rgb01(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

def rgb01_to_hex(rgb):
    rgb = np.clip(np.array(rgb) * 255.0, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def interpolate_hex_palette(base_hex_colors, n_out):
    """Interpole linéairement une palette hex (RGB) vers n_out couleurs."""
    base = np.stack([hex_to_rgb01(c) for c in base_hex_colors], axis=0)  # [N,3]
    x_base = np.linspace(0.0, 1.0, len(base_hex_colors))
    x_out = np.linspace(0.0, 1.0, n_out)
    out = np.stack([np.interp(x_out, x_base, base[:, ch]) for ch in range(3)], axis=1)  # [n_out,3]
    return [rgb01_to_hex(out[i]) for i in range(n_out)]

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def display_latent(labels: torch.Tensor, 
               mu_logvars: None|tuple[torch.Tensor]=None,
               z: None|torch.Tensor=None,
               title: str="latent space", 
               test: bool=False,
               norm=None,) :
    """
    Generate a plot to visualize in 2D the latent space.
    Ensure that if z=None, mu_logvars is not None.
    
    Args:
    feats: tuple[torch.Tensor], ((number_sample,latent_dim), (number_sample,latent_dim))
    labels: torch.Tensor, (number_sample, 1)
    z: None|torch.Tensor, (number_sample, latent_dim). Allows to give directly the latent vector. 
    test: bool, if inference set True.
    """
    assert (z is not None and mu_logvars is None) or (z is None and mu_logvars is not None), "Among z and mu_logvars, one should be at None. Both can't be."

    pca = PCA(n_components=2)
    latent = []

    unique_labels = labels.unique(sorted=True).detach().cpu().numpy()
    K = len(unique_labels)

    # --- palettes
    colors10 = ["red","orange","yellow","lightgreen","green","lightblue","darkblue","royalblue","purple","pink"]

    base50 = [
        '#ff0000', '#ff1f00', '#ff3d00', '#ff5c00', '#ff7a00',
        '#ff9900', '#ffb800', '#ffd600', '#fff500', '#ebff00',
        '#ccff00', '#aeff00', '#8fff00', '#70ff00', '#52ff00',
        '#33ff00', '#14ff00', '#00ff0a', '#00ff29', '#00ff47',
        '#00ff66', '#00ff85', '#00ffa3', '#00ffc2', '#00ffe0',
        '#00ffff', '#00e0ff', '#00c2ff', '#00a3ff', '#0085ff',
        '#0066ff', '#0047ff', '#0029ff', '#000aff', '#1400ff',
        '#3300ff', '#5200ff', '#7000ff', '#8f00ff', '#ae00ff',
        '#cc00ff', '#eb00ff', '#ff00f5', '#ff00d6', '#ff00b8',
        '#ff0099', '#ff007a', '#ff005c', '#ff003d', '#ff001f'
    ]

    # --- choix cmap
    if K == 10:
        cmap = ListedColormap(colors10, name="mycats10")
        print("\ncmap : personalised (10)\n")

    elif K < 10:
        # tab10 discret mais on garde un norm basé sur tes labels réels
        cmap = plt.get_cmap("tab10", K)

    elif K <= 50:
        cmap = ListedColormap(base50[:K], name="mycats50")

    elif K <= 200:
        colors200 = interpolate_hex_palette(base50, 200)
        cmap = ListedColormap(colors200[:K], name="mycats200")

    else:
        # fallback si jamais >200
        cmap = plt.get_cmap("hsv", K)

    # --- norm/bounds (important si labels ne sont pas 0..K-1)
    bounds = np.concatenate([unique_labels - 0.5, [unique_labels[-1] + 0.5]])
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
            

    if z is None :
        z = sample_from(mu_logvars, test=test)

    #pca 
    latent_pca = z.detach().cpu().numpy()
    explained_axis = [-1, -1]
    if not z.shape[1] in [1,2] : 
        latent_pca = pca.fit_transform(latent_pca)
        explained_axis = pca.explained_variance_ratio_

    #variance explained : if -2, no pca has been run
    explained = np.sum(explained_axis)
    pts = latent_pca

    if z.shape[1] == 1 : pts_y = np.zeros_like(pts[:, 0])
    else : pts_y = pts[:, 1]
    pts_x = pts[:, 0]

    pts_x= del_outliers(arr=pts_x, k=5)
    pts_y = del_outliers(arr=pts_y, k=5)
    
    plt.scatter(pts_x, pts_y, c=labels.squeeze(1), cmap=cmap, alpha=0.4, norm=norm)
    plt.xlabel(f"{explained_axis[0]}")
    plt.ylabel(f"{explained_axis[1]}")
    #cbar = plt.colorbar(sc, ticks=unique_labels)
    #cbar.ax.set_yticklabels([f'class {int(c)}' for c in unique_labels])

    plt.title(title+f"-explained : {explained}")
    plt.grid()
    plt.show()
    

def set_device(pref_gpu: int=0, verb=False) -> str :
    """
    Looking for a GPU and display informations if available.

    Args:
        pref_gpu: int, id of the main gpu.
    Return:
        device: str, name of device (cpu or cuda)
    """
    is_gpu = torch.cuda.is_available()

    device = f"cuda:{pref_gpu}" if is_gpu else "cpu"

    if is_gpu :
        if verb : print("Nombre de GPU :", torch.cuda.device_count())
        if pref_gpu>=torch.cuda.device_count() : 
            pref_gpu = 0
            if verb : print(f"{pref_gpu} gpu is not available. Switched on gpu0")

        for i in range(torch.cuda.device_count()):
            if verb:
                print(f"\n[ GPU {i} ]")
                print("Nom :", torch.cuda.get_device_name(i))
                print("Mémoire totale :", round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2), "Go")
                print("Mémoire utilisée :", round(torch.cuda.memory_allocated(i) / 1e9, 2), "Go")
                print("Mémoire réservée :", round(torch.cuda.memory_reserved(i) / 1e9, 2), "Go")

    print(f"current device is {torch.cuda.current_device()}")
    return device, is_gpu
 

############### merge plots
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def merge_images(*image_paths, labels=None, font_size=20, separator_height=10):
    """
    Merge un nombre arbitraire d'images verticalement, avec un titre centré au-dessus de chacune.

    Utilisation :
        - merge_images(p1, p2, p3, labels=["gen", "s", "t"])
        - merge_images([p1, p2, p3], labels=[...])  # liste unique aussi gérée
    """

    # Autoriser un appel de type merge_images([p1, p2, p3], labels=...)
    if len(image_paths) == 1 and isinstance(image_paths[0], (list, tuple)):
        image_paths = list(image_paths[0])

    if labels is None:
        # par défaut : nom du fichier sans extension
        labels = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    assert len(image_paths) == len(labels), "len(labels) doit == len(image_paths)"

    images = [Image.open(path) for path in image_paths]
    widths = [img.width for img in images]
    assert all(w == widths[0] for w in widths), "Toutes les images doivent avoir la même largeur"

    # Police
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text_height = font_size + 10

    # Séparateur noir entre les blocs
    separator = Image.fromarray(
        np.zeros((separator_height, widths[0], 3), dtype=np.uint8)
    )

    final_parts = []
    for i, (img, label) in enumerate(zip(images, labels)):
        # Bandeau de texte
        text_img = Image.new("RGB", (widths[0], text_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        text_width = draw.textlength(label, font=font)
        draw.text(
            ((widths[0] - text_width) // 2, 5),
            label,
            fill=(255, 255, 255),
            font=font,
        )

        final_parts.append(text_img)
        final_parts.append(img)

        if i != len(images) - 1:
            final_parts.append(separator)

    total_height = sum(p.height for p in final_parts)
    final_image = Image.new("RGB", (widths[0], total_height))

    y = 0
    for part in final_parts:
        final_image.paste(part, (0, y))
        y += part.height

    return final_image


def merge_images_with_black_gap(image_paths, gap=10):
    images = [Image.open(p) for p in image_paths]
    widths = [img.width for img in images]
    if len(set(widths)) != 1:
        raise ValueError("Toutes les images doivent avoir la même largeur")
    W = widths[0]
    separator = Image.new("RGB", (W, gap), color=(0, 0, 0))
    parts = []
    for img in images[:-1]:
        parts.append(img)
        parts.append(separator)
    parts.append(images[-1])
    total_h = sum(p.height for p in parts)
    merged = Image.new("RGB", (W, total_h), color=(0, 0, 0))
    y = 0
    for p in parts:
        merged.paste(p, (0, y))
        y += p.height
    return merged

def grid_merge(image_paths, out_path, cols=3, padding=10, bg=(0,0,0), resize_to=None):
    """
    Merge 6 images en une grille 2x3 (ou plus généralement rows x cols).
    - image_paths: liste de 6 chemins d’images.
    - cols: nb de colonnes (3 pour ton cas).
    - padding: espace en pixels entre les images.
    - bg: couleur de fond (R,G,B).
    - resize_to: (W,H) optionnel pour forcer une taille identique avant collage.
    """
    assert len(image_paths) > 0
    imgs = [Image.open(p).convert("RGB") for p in image_paths]

    # Option : harmoniser les tailles
    if resize_to is not None:
        imgs = [im.resize(resize_to, Image.BICUBIC) for im in imgs]
    else:
        # par défaut on prend la taille de la plus petite (évite les débords)
        w = min(im.width for im in imgs)
        h = min(im.height for im in imgs)
        imgs = [im.resize((w, h), Image.BICUBIC) for im in imgs]

    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)

    W = cols * w + (cols - 1) * padding
    H = rows * h + (rows - 1) * padding
    canvas = Image.new("RGB", (W, H), bg)

    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x = c * (w + padding)
        y = r * (h + padding)
        canvas.paste(im, (x, y))

    canvas.save(out_path)
    return canvas


#########Log for models
def log_cross_cov_heatmap(mu_s, logvar_s, mu_t, logvar_t, save_path: str, interactive: bool=False):
    cov_mu = cross_cov(mu_s, mu_t).detach().cpu().numpy()
    assert cov_mu.shape == (mu_s.shape[1], mu_t.shape[1]), "ERROR COV MATRIX SHAPE"
    cov_logvar = cross_cov(logvar_s, logvar_t).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(cov_mu, ax=axes[0], cmap="coolwarm", center=0, cbar=True,)
    axes[0].set_title("cross_cov(mu_s, mu_t)")

    sns.heatmap(cov_logvar, ax=axes[1], cmap="coolwarm", center=0, cbar=True)
    axes[1].set_title("cross_cov(logvar_s, logvar_t)")

    plt.tight_layout()
    if interactive : plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def report_nonfinite(**named_tensors):
    problems = []
    for name, t in named_tensors.items():
        if not torch.is_tensor(t): 
            continue
        mask = ~torch.isfinite(t)
        if mask.any():
            n_nan = torch.isnan(t).sum().item()
            n_inf = torch.isinf(t).sum().item()
            # premiers indices problématiques (max 5)
            ex = mask.nonzero(as_tuple=False)[:5].tolist()
            problems.append(f"- {name}: NaN={n_nan}, Inf={n_inf}, ex_idx={ex}")
    if problems:
        raise RuntimeError("Non-finite detected:\n" + "\n".join(problems))
    