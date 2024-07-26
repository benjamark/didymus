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
npys_dir = f"{project_dir}/npys_ext"

# load SDFs
sdfs = [ np.load(f"{project_dir}/corner_0.npy"),np.load(f"{project_dir}/corner_1.npy"),np.load(f"{project_dir}/corner_2.npy") ]


# load barycentric coordinates
w1 = np.load(f'{project_dir}/w1_ext.npy')
w2 = np.load(f'{project_dir}/w2_ext.npy')
w3 = np.load(f'{project_dir}/w3_ext.npy')


# create necessary directories
os.makedirs(f"{npys_dir}", exist_ok=True)

# get the sample index from the argument
sample_idx = int(sys.argv[1])

# process the specified sample
interpolated_sdf = interpolate_sdfs_2d(sdfs, w1[sample_idx], w2[sample_idx], w3[sample_idx])
#np.save('sdf.npy', interpolated_sdf)  ## REMOVE THIS!!!
interface = (interpolated_sdf >= -epsilon) & (interpolated_sdf <= epsilon)
scalar_field = interface.astype(np.float32)

# extract interface using marching cubes
vertices, faces, normals, values = measure.marching_cubes(scalar_field, level=0.5)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
np.save(f'{npys_dir}/{sample_idx}.npy', interface)
