import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as distance
from skimage import measure
import trimesh
import os
import sys
from helpers import load_config
import time

def interpolate_sdfs_2d(sdfs, w1, w2, w3):
    return w1*sdfs[0] + w2*sdfs[1] + w3*sdfs[2]

# load the configuration file
config = load_config()
corner_stls = config["corner_stls"]
resolution = config["resolution"]
project_dir = config["project_dir"]
samples_per_dim = config["samples_per_dim"]
epsilon = config["epsilon"]
project_dir = f'{project_dir}_r{resolution}_n{samples_per_dim}'
npys_dir = f"{project_dir}/npys"

# load SDFs
sdfs = np.load(f"{project_dir}/sdfs.npy")

# load barycentric coordinates
w1 = np.load(f'{project_dir}/w1.npy')
w2 = np.load(f'{project_dir}/w2.npy')
w3 = np.load(f'{project_dir}/w3.npy')

# create necessary directories
os.makedirs(f"{npys_dir}", exist_ok=True)

# get the sample index from the argument
sample_idx = int(sys.argv[1])

# process the specified sample
interpolated_sdf = interpolate_sdfs_2d(sdfs, w1[sample_idx], w2[sample_idx], w3[sample_idx])
interface = (interpolated_sdf >= -epsilon) & (interpolated_sdf <= epsilon)
np.save(f'{npys_dir}/{sample_idx}.npy', interface)
