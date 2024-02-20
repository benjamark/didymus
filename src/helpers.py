import numpy as np
import json

def compute_bbox(stl_mesh, buffer_percent=5):
    """
    Compute the bounding box of an STL file with a buffer on all sides.
    """
    # reshape the points array to a (n, 3, 3) array 
    reshaped_points = stl_mesh.points.reshape(-1, 3, 3)

    # find the min and max points
    min_point = np.min(reshaped_points, axis=(0, 1))
    max_point = np.max(reshaped_points, axis=(0, 1))

    # calculate the buffer size
    buffer_size = ((max_point - min_point) * buffer_percent / 100)

    # apply the buffer to the bounding box
    x_min, y_min, z_min = min_point - buffer_size
    x_max, y_max, z_max = max_point + buffer_size

    return x_min, x_max, y_min, y_max, z_min, z_max


def center_bbox(stl_mesh):
    """
    Center the mesh at (0,0,0) by moving the bounding box center
    """
    x_min, x_max, y_min, y_max, z_min, z_max = compute_bbox(stl_mesh, 0)

    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    center_z = (z_max + z_min) / 2

    for i in range(len(stl_mesh.vectors)):
        for j in range(3):  # Each triangle has 3 vertices
            stl_mesh.vectors[i][j][0] -= center_x
            stl_mesh.vectors[i][j][1] -= center_y
            stl_mesh.vectors[i][j][2] -= center_z

    return stl_mesh


def dump_rays_to_file(rays, filename):
    with open(filename, 'w') as file:
        for ray in rays:
            ray_origin = ray['origin']
            ray_direction = ray['direction']
            file.write(f"Origin: {ray_origin}, Direction: {ray_direction}\n")


def load_config(config_file='config.json'):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config
