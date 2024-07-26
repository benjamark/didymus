import numpy as np
import pymeshlab
import trimesh
from skimage import measure
from helpers import load_config
import os
import sys

config = load_config()
smooth_iter = config["smooth_iter"]
resolution = config["resolution"]
samples_per_dim = config["samples_per_dim"]
project_dir = config["project_dir"]

project_dir = f'{project_dir}_r{resolution}_n{samples_per_dim}'
input_directory = f'{project_dir}/npys_ext/'
output_directory = f'{project_dir}/stls_ext/'

os.makedirs(output_directory, exist_ok=True)

epsilon = 0.2

index = int(sys.argv[1])
npy_file = f"{index}.npy"
npy_path = os.path.join(input_directory, npy_file)

# checks for extant stls before creating stl
smooth_stl_path = os.path.join(output_directory, npy_file.replace('.npy', '.stl'))
if os.path.exists(smooth_stl_path):
    sys.exit()

bin_npy = np.load(npy_path)

interface = (bin_npy >= 1.0 - epsilon) & (bin_npy <= 1.0 + epsilon)
scalar_field = interface.astype(np.float32)

# extract interface using marching cubes
vertices, faces, normals, values = measure.marching_cubes(scalar_field, level=0.5)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# export to a temporary STL file
temp_stl_path = os.path.join(output_directory, f'recon_{index}.stl')
mesh.export(temp_stl_path)

ms = pymeshlab.MeshSet()
ms.load_new_mesh(temp_stl_path)

ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=smooth_iter)

smooth_stl_path = os.path.join(output_directory, npy_file.replace('.npy', '.stl'))
ms.save_current_mesh(smooth_stl_path)

# delete the temporary STL file
os.remove(temp_stl_path)

