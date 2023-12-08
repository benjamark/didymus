import numpy as np
from stl import mesh
from helpers.intersects import ray_intersects_triangle

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.direction /= np.linalg.norm(self.direction)



stl = mesh.Mesh.from_file('../utils/sphere.stl')

# mesh bounds (TODO: compute bounding box of STL to determine these)
x_min, x_max = -1.2, 1.2
y_min, y_max = -1.2, 1.2
z_min, z_max = -1.2, 1.2

resolution = 100  

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

print(len(rays))
