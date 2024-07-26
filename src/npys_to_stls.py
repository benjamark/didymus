import numpy as np
import pymeshlab
import trimesh
from skimage import measure
from helpers import load_config
import os

config = load_config()
smooth_iter = config["smooth_iter"]
resolution = config["resolution"]
samples_per_dim = config["samples_per_dim"]
project_dir = config["project_dir"]

project_dir = f'{project_dir}_r{resolution}_n{samples_per_dim}'
input_directory = f'{project_dir}/npys/'
output_directory = f'{project_dir}/stls/'

os.makedirs(output_directory, exist_ok=True)

npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]

epsilon = 0.2

for i,npy_file in enumerate(npy_files):
    # checks for extant stls before creating stl
    print(i)
    smooth_stl_path = os.path.join(output_directory, npy_file.replace('.npy', '.stl'))
    if os.path.exists(smooth_stl_path):
        continue

    npy_path = os.path.join(input_directory, npy_file)
    
    bin_npy = np.load(npy_path)#[0,:,:,:]

    interface = (bin_npy >= 1.0 - epsilon) & (bin_npy <= 1.0 + epsilon)
    scalar_field = interface.astype(np.float32)
    
    # extract interface using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(scalar_field, level=0.5)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # export to a temporary STL file
    temp_stl_path = os.path.join(output_directory, 'recon.stl')
    mesh.export(temp_stl_path)
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(temp_stl_path)
    
    ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=smooth_iter)
    
    smooth_stl_path = os.path.join(output_directory, npy_file.replace('.npy', '.stl'))
    ms.save_current_mesh(smooth_stl_path)
    
    # delete the temporary STL file
    os.remove(temp_stl_path)
