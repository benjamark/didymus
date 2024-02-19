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
barycentric_coords = generate_barycentric_coordinates(num_stls - 1, samples_per_dim)

npys_dir = f"{project_dir}/npys"
sdfs_dir = f"{project_dir}/sdfs"
os.makedirs(f"{npys_dir}", exist_ok=True) 
os.makedirs(f"{sdfs_dir}", exist_ok=True) 

# process each combination of barycentric coordinates
count = 0
for coords in barycentric_coords:
    print(count)
    interpolated_sdf = interpolate_sdfs(sdfs, coords)
    
    interface = (interpolated_sdf >= -epsilon) & \
                (interpolated_sdf <= epsilon)

    scalar_field = interface.astype(np.float32)
    # extract interface using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(scalar_field, level=0.5)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # export mesh
    if mesh.is_watertight:
        coords_str = '_'.join(f"{int(coord * 100)}" for coord in coords)
        filename = f"{sdfs_dir}/sdf_{coords_str}.stl"
        # only write watertight stls
        mesh.export(filename)
        # only write corresponding numpy arrs
        np.save(f'{npys_dir}/{count}.npy', interface)
    else:
        with open(f'{project_dir}/leaky_coords.txt', 'a') as file:
            coord_str = ' '.join(map(str, coords))  
            file.write(coord_str + '\n')  

    print(f'Watertightness check: {mesh.is_watertight}')
    count += 1
