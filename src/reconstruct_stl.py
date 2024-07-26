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
    """
    Generate all possible combinations of 2D barycentric weights (w1, w2, w3)
    with a specified resolution and save them to numpy arrays.
    """
    w1_list, w2_list, w3_list = [], [], []
    step = 1 / resolution

    for w1 in range(resolution + 1):
        for w2 in range(resolution + 1 - w1):
            w3 = resolution - w1 - w2
            w1_list.append(w1 * step)
            w2_list.append(w2 * step)
            w3_list.append(w3 * step)

    w1_array = np.array(w1_list)
    w2_array = np.array(w2_list)
    w3_array = np.array(w3_list)

    return w1_array, w2_array, w3_array


def interpolate_sdfs_2d(sdfs, w1, w2, w3):
    return w1*sdfs[0] +w2*sdfs[1] +w3*sdfs[2]


sdfs = [binary_fill_holes(np.load(f"{project_dir}/corner_{i}.npy")) for i in range(len(corner_stls))]
t0 = time.time()
sdfs = [distance(~sdf) - distance(sdf) for sdf in sdfs]

np.save('sdfs.npy', sdfs)
print(sdfs.shape)
breakpoint()

t1 = time.time()
print(f'Timings :: SDF (distance transform): {t1-t0} (s)')

# generate barycentric coordinates
num_stls = len(config["corner_stls"])

w1, w2, w3 = gen_2d_bary_weights(samples_per_dim)
np.save(f'{project_dir}/w1.npy', w1)
np.save(f'{project_dir}/w2.npy', w2)
np.save(f'{project_dir}/w3.npy', w3)

npys_dir = f"{project_dir}/npys"
sdfs_dir = f"{project_dir}/sdfs"
os.makedirs(f"{npys_dir}", exist_ok=True) 
os.makedirs(f"{sdfs_dir}", exist_ok=True) 

# process each combination of barycentric coordinates
count = 0
print(f'Total samples: {w1.shape}')
for i in range(w1.shape[0]):
    print(count)
    interpolated_sdf = interpolate_sdfs_2d(sdfs, w1[i], w2[i], w3[i])
    
    interface = (interpolated_sdf >= -epsilon) & \
                (interpolated_sdf <= epsilon)

    scalar_field = interface.astype(np.float32)
    # extract interface using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(scalar_field, level=0.5)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # export mesh
    # NOTE: WE WRITE ALL STLS REGARDLESS OF WATERTIGHTNESS BECAUSE OF 
    # only write corresponding numpy arrs
    np.save(f'{npys_dir}/{i}.npy', interface)
    print(i)
    count += 1
