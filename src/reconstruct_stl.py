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


def gen_nd_bary_weights(resolution, n):
    """
    Generate barycentric weights for an n-dimensional simplex using a nifty
    recursive implementation.

    Computes all possible combinations of barycentric weights
    for an n-dimensional simplex, given a specified resolution.

    Parameters
    ----------
    resolution : int
        No. of divisions along each axis of the simplex.

    n : int
        The dimension of the simplex. e.g., `n=2` for a triangle (2-simplex),
        `n=3` for a tetrahedron (3-simplex), etc.

    Returns
    -------
    numpy.ndarray
        2D numpy array. Each row represents a set of barycentric weights for a
        point within the n-dimensional simplex.

    """
    def generate_weights(current, level):
        if level == n:  # base case: last weight
            return [current + [resolution - sum(current)]]
        else:
            combinations = []
            for i in range(resolution - sum(current) + 1):
                combinations += generate_weights(current + [i], level + 1)
            return combinations

    raw_combinations = generate_weights([], 0)
    scaled_combinations = np.array(raw_combinations) / resolution
    return scaled_combinations


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

def generate_barycentric_coordinates(num_dimensions, steps):
    # generate all combinations with replacement to ensure all sums are <= 1
    comb = combinations_with_replacement(range(steps), num_dimensions)
    valid_coords = [np.array(c) / (steps - 1) for c in comb if sum(c) <= (steps - 1)]
    # write coordinates to file
    with open(f'{project_dir}/valid_coords.txt', 'w') as file:
        for coord in valid_coords:
            coord_str = ' '.join(map(str, coord))  
            file.write(coord_str + '\n')  
    
    return valid_coords

def interpolate_sdfs(sdfs, coords):
    interpolated_sdf = np.zeros_like(sdfs[0])
    for i, coord in enumerate(coords):
        interpolated_sdf += coord * sdfs[i]
    interpolated_sdf += (1 - sum(coords)) * sdfs[-1]
    return interpolated_sdf

sdfs = [binary_fill_holes(np.load(f"{project_dir}/corner_{i}.npy")) for i in range(len(corner_stls))]
t0 = time.time()
sdfs = [distance(~sdf) - distance(sdf) for sdf in sdfs]
t1 = time.time()
print(f'Timings :: SDF (distance transform): {t1-t0} (s)')

# generate barycentric coordinates
num_stls = len(config["corner_stls"])
#barycentric_coords = generate_barycentric_coordinates(num_stls - 1, samples_per_dim)

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
    #if mesh.is_watertight:
    filename = f"{sdfs_dir}/sdf_{i}.stl"
    # only write watertight stls
    #mesh.export(filename)
    # write sdfs instead
    np.save(f'{sdfs_dir}/{i}.npy', interpolated_sdf)
    # only write corresponding numpy arrs
    np.save(f'{npys_dir}/{i}.npy', interface)

    print(i)
    print(f'Watertightness check: {mesh.is_watertight}')
    count += 1

