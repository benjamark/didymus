import pymeshlab
import os
import trimesh

input_directory = './sdfs/'
output_directory = './ssdfs/'

stl_files = [f for f in os.listdir(input_directory) if f.endswith('.stl')]

for stl_file in stl_files:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(input_directory, stl_file))

    ms.apply_filter('apply_coord_laplacian_smoothing', stepsmoothnum=5)

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