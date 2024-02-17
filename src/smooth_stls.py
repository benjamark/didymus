import pymeshlab
import os
import trimesh
from helpers import load_config

config = load_config()
smooth_iter = config["smooth_iter"]

# TODO: make relative to project_dir
input_directory = 'npys/images_r257/sdfs/'
output_directory = 'npys/images_r257/ssdfs/'

stl_files = [f for f in os.listdir(input_directory) if f.endswith('.stl')]

for stl_file in stl_files:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(input_directory, stl_file))

    ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=smooth_iter)

    temp_file = 'temp.stl'
    ms.save_current_mesh(temp_file)

    tri_mesh = trimesh.load(temp_file)
    if tri_mesh.is_watertight:
        print(f"{stl_file} is watertight after smoothing.")
    else:
        print(f"{stl_file} is NOT watertight after smoothing.")

    output_filename = stl_file.replace('.stl', '_smoothed.stl')
    ms.save_current_mesh(os.path.join(output_directory, output_filename))

    os.remove(temp_file)
