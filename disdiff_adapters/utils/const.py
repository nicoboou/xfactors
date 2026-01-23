from os.path import join
from os import getenv
from dataclasses import dataclass
from typing import ClassVar

#General Const
PROJECT_PATH= "."

LOG_DIR = join(PROJECT_PATH, "disdiff_adapters/logs")

#Const by Dataset/Model
@dataclass
class Shapes3D :

    @dataclass
    class Path :
        H5 = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/3dshapes.h5")
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_train.npz")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_val.npz")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/shapes3d_test.npz")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/split_3dshapes.npz")
        BUFF_IMG = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/images_train_buff.pt")
        BUFF_LABELS = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/labels_train_buff.pt")
    @dataclass
    class Params :
        FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                        'orientation']
        
        NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                            'scale': 8, 'shape': 4, 'orientation': 15}
        
@dataclass
class MPI3D :

    @dataclass
    class Path :
        H5 = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/3dshapes.h5")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/mpi3d/mpi3d_toy.npz")
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/mpi3d/mpi3d_train.npz")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/mpi3d/mpi3d_val.npz")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/mpi3d/mpi3d_test.npz")
    @dataclass
    class Params :
        FACTORS_IN_ORDER = ['object_color', 'object_shape', 'object_size', 'camera_height', 'background_color',
                        'horizontal_axis', "vertical_axis"]
        
        NUM_VALUES_PER_FACTOR = {'object_color': 6, 'object_shape': 6, 'object_size': 2, 
                            'camera_height': 3, 'background_color': 3, 'horizontal_axis': 40,
                            "vertical_axis":40}

@dataclass
class BloodMNIST :
    
    @dataclass
    class Path :
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist_train.pt")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist_val.pt")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist_test.pt")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/bloodmnist/bloodmnist.npz")
        H5 = join(PROJECT_PATH,"disdiff_adapters/data/bloodmnist/bloodmnist.h5" )
        VAE = join(LOG_DIR, "vae/bloodmnist")

    # @dataclass
    # class Params :
    #     N_TRAIN = 
    #     N_VAL =
    #     N_TEST = 

@dataclass
class CelebA :

    @dataclass
    class Path :
        DATA = "/projects/compures/alexandre/PyTorch-VAE/Data/"
        BUFF_IMG = join(PROJECT_PATH, "disdiff_adapters/data/celeba/images_train_buff.pt")
        BUFF_LABELS = join(PROJECT_PATH, "disdiff_adapters/data/celeba/labels_train_buff.pt")
        ## None
        H5 = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/3dshapes.h5")
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/celeba/celeba_train.npz")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/celeba/celeba_val.npz")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/celeba/celeba_test.npz")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/split_3dshapes.npz")
        BUFF_IMG = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/images_train_buff.pt")
        BUFF_LABELS = join(PROJECT_PATH, "disdiff_adapters/data/3dshapes/labels_train_buff.pt")

    class Params : 
        FACTORS_IN_ORDER = [
        "5_o_Clock_Shadow",   # 0
        "Arched_Eyebrows",    # 1
        "Attractive",         # 2
        "Bags_Under_Eyes",    # 3
        "Bald",               # 4
        "Bangs",              # 5
        "Big_Lips",           # 6
        "Big_Nose",           # 7
        "Black_Hair",         # 8
        "Blond_Hair",         # 9
        "Blurry",             # 10
        "Brown_Hair",         # 11
        "Bushy_Eyebrows",     # 12
        "Chubby",             # 13
        "Double_Chin",        # 14
        "Eyeglasses",         # 15
        "Goatee",             # 16
        "Gray_Hair",          # 17
        "Heavy_Makeup",       # 18
        "High_Cheekbones",    # 19
        "Male",               # 20
        "Mouth_Slightly_Open",# 21
        "Mustache",           # 22
        "Narrow_Eyes",        # 23
        "No_Beard",           # 24
        "Oval_Face",          # 25
        "Pale_Skin",          # 26
        "Pointy_Nose",        # 27
        "Receding_Hairline",  # 28
        "Rosy_Cheeks",        # 29
        "Sideburns",          # 30
        "Smiling",            # 31
        "Straight_Hair",      # 32
        "Wavy_Hair",          # 33
        "Wearing_Earrings",   # 34
        "Wearing_Hat",        # 35
        "Wearing_Lipstick",   # 36
        "Wearing_Necklace",   # 37
        "Wearing_Necktie",    # 38
        "Young",              # 39
    ]     
    
        DISEN_BASE_IDX = [
        "5_o_Clock_Shadow",   # 0
        "Arched_Eyebrows",    # 1
        "Attractive",         # 2
        "Bags_Under_Eyes",    # 3
        "Bald",               # 4
        "Bangs",              # 5
        "Big_Lips",           # 6
        "Big_Nose",           # 7
        "Black_Hair",         # 8
        "Blond_Hair",         # 9
        "Blurry",             # 10
        "Brown_Hair",         # 11
        "Bushy_Eyebrows",     # 12
        "Chubby",             # 13
        "Double_Chin",        # 14
        "Eyeglasses",         # 15
        "Goatee",             # 16
        "Gray_Hair",          # 17
        "Heavy_Makeup",       # 18
        "High_Cheekbones",    # 19
        "Male",               # 20
        "Mouth_Slightly_Open",# 21
        "Mustache",           # 22
        "Narrow_Eyes",        # 23
        "No_Beard",           # 24
        "Oval_Face",          # 25
        "Pale_Skin",          # 26
        "Pointy_Nose",        # 27
        "Receding_Hairline",  # 28
        "Rosy_Cheeks",        # 29
        "Sideburns",          # 30
        "Smiling",            # 31
        "Straight_Hair",      # 32
        "Wavy_Hair",          # 33
        "Wearing_Earrings",   # 34
        "Wearing_Hat",        # 35
        "Wearing_Lipstick",   # 36
        "Wearing_Necklace",   # 37
        "Wearing_Necktie",    # 38
        "Young",              # 39
        ]

        REPRESENTANT = [
        "Eyeglasses",         # 15
        "Male",               # 20
        "Pale_Skin",          # 26
        "Smiling",            # 31
        "Wearing_Hat",        # 35
        "Wearing_Lipstick",   # 36
    ]

        REPRESENTANT_IDX = [15, 20, 26, 31, 35, 36]

@dataclass
class MNIST:
    @dataclass
    class Path :
        data_dir: str = join(PROJECT_PATH, "disdiff_adapters/data/mnist")

@dataclass
class DSprites:
    @dataclass
    class Path :
        H5 = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites.h5")
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites_train.npz")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites_val.npz")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites_test.npz")
        NPZ = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/dsprites.npz")
        BUFF_IMG = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/images_train_buff.pt")
        BUFF_LABELS = join(PROJECT_PATH, "disdiff_adapters/data/dsprites/labels_train_buff.pt")

    @dataclass
    class Params :
        FACTORS_IN_ORDER = ['shape', 'scale', 'orientation', 'pos_x',
                        'pos_y']
        
        NUM_VALUES_PER_FACTOR = {'shape': 3, 'scale': 6, 
                            'orientation': 40, 'pos_x': 32, 'pos_y': 32}
        
@dataclass
class Cars3D:
    @dataclass
    class Path :
        CACHE = join(PROJECT_PATH, "disdiff_adapters/data/cars3d_cache/")
        LOCAL = join(PROJECT_PATH, "disdiff_adapters/data/cars3d")
        TRAIN = join(PROJECT_PATH, "disdiff_adapters/data/cars3d/cars3d_train.npz")
        VAL = join(PROJECT_PATH, "disdiff_adapters/data/cars3d/cars3d_val.npz")
        TEST = join(PROJECT_PATH, "disdiff_adapters/data/cars3d/cars3d_test.npz")
        H5 = join(PROJECT_PATH, "disdiff_adapters/data/cars3d/cars3d_train.npz")

    @dataclass
    class Params :
        FACTORS_IN_ORDER = ["identity", "elevation_angle", "azimuth_angle"]
        
        NUM_VALUES_PER_FACTOR = {"identity": 183, "elevation_angle": 4, "azimuth_angle": 24}