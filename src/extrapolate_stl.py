import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as distance
from skimage import measure
import trimesh
import os
from itertools import combinations_with_replacement
from helpers import load_config
import time

# load the configuration file
config = load_config()

corner_stls = config["corner_stls"]
resolution = config["resolution"]
project_dir = config["project_dir"]
samples_per_dim = config["samples_per_dim"]
epsilon = config["epsilon"]
project_dir = f'{project_dir}_r{resolution}_n{samples_per_dim}'
print(project_dir)


def gen_2d_bary_weights(resolution):

    w1_list, w2_list, w3_list = [], [], []
    step = 1 / resolution

    for w1 in np.arange(0, 1 + step, step):
        for w2 in np.arange(0, 1 + step, step):
            for w3 in np.arange(0, 1 + step, step):
                if (w1 +w2 +w3 > 2.0):
                    w1_list.append(w1)
                    w2_list.append(w2)
                    w3_list.append(w3)

    w1_array = np.array(w1_list)
    w2_array = np.array(w2_list)
    w3_array = np.array(w3_list)
    return w1_array, w2_array, w3_array

# generate barycentric coordinates
num_stls = len(config["corner_stls"])

w1, w2, w3 = gen_2d_bary_weights(samples_per_dim)
np.save(f'{project_dir}/w1_ext.npy', w1)
np.save(f'{project_dir}/w2_ext.npy', w2)
np.save(f'{project_dir}/w3_ext.npy', w3)

