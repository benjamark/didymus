import numpy as np
from stl import mesh
from helpers.ray_tracing import Ray, ray_intersects_triangle

def get_cell_index(point):
    """ Given an intersection point, identify the cell indices in the mesh
        it should be assigned to
    """
    x, y, z = point
    x_index = int((x - x_min) / (x_max - x_min) * resolution)
    y_index = int((y - y_min) / (y_max - y_min) * resolution)
    z_index = int((z - z_min) / (z_max - z_min) * resolution)
    return x_index, y_index, z_index


stl = mesh.Mesh.from_file('../utils/sphere.stl')

# mesh bounds (TODO: compute bounding box of STL to determine these)
x_min, x_max = -1.2, 1.2
y_min, y_max = -1.2, 1.2
z_min, z_max = -1.2, 1.2

resolution = 10

x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
z = np.linspace(z_min, z_max, resolution)
# mesh
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# mesh to record intersections (only x for now)
intersection_grid = np.zeros((resolution, resolution, resolution), dtype=np.bool)

# define x-rays
rays = []
for y in range(resolution):
    for z in range(resolution):
        # rays originate from face-centers, at leftmost face 
        ray_origin = [x_min, \
                      y_min + (y + 0.5) * (y_max - y_min) / resolution, \
                      z_min + (z + 0.5) * (z_max - z_min) / resolution]
        ray_direction = [1., 0., 0.]  # along the x-axis
        rays.append(Ray(ray_origin, ray_direction))

# ray-tracing in x
count = 0
for ray in rays:
    print(count)
    count = count+1
    first_intersection = None
    last_intersection = None

    for triangle in stl.vectors:
        intersects, point = ray_intersects_triangle(ray, triangle)
        if intersects:
            if first_intersection is None:
                first_intersection = point
            last_intersection = point

    if first_intersection is not None:
        print('in')
        x_index, y_index, z_index = get_cell_index(first_intersection)
        intersection_grid[x_index, y_index, z_index] = True  # Mark entry cell

    if last_intersection is not None:
        print('out')
        x_index, y_index, z_index = get_cell_index(last_intersection)
        intersection_grid[x_index, y_index, z_index] = True  # Mark exit cell


import matplotlib.pyplot as plt

x_index = 2

plt.imshow(intersection_grid[x_index, :, :], cmap='gray')
plt.xlabel('Z index')
plt.ylabel('Y index')
plt.show()
