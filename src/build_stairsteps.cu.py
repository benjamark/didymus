import numpy as np
from stl import mesh
from numba import cuda
from kernels import ray_intersects_tri, get_grid_index, trace_rays

# load test STL
sphere_mesh = mesh.Mesh.from_file('../utils/sphere.stl')
# ensure triangles are numpy arrays for GPU usage
tris = np.array(sphere_mesh.vectors, dtype=np.float32)

# define ray data type; use structured numpy array for GPU usage
Ray = np.dtype([('origin',    np.float32, (3,)), 
                ('direction', np.float32, (3,))])


# create background structured mesh
x_min, x_max = -1.2, 1.2
y_min, y_max = -1.2, 1.2
z_min, z_max = -1.2, 1.2
resolution = 100

x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
z = np.linspace(z_min, z_max, resolution)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# mesh to record intersections
intersection_grid = np.zeros((resolution, resolution, resolution), dtype=np.bool_)

# Transfer to GPU
X_ = cuda.to_device(X)
Y_ = cuda.to_device(Y)
Z_ = cuda.to_device(Z)
intersection_grid_ = cuda.to_device(intersection_grid)

# initialize a list to store ray data
ray_data = []
# initialize x-rays
for y in range(resolution):
    for z in range(resolution):
        # rays originate at y/z face centers
        ray_origin = [x_min,
                      y_min + (y + 0.5) * (y_max - y_min) / resolution,
                      z_min + (z + 0.5) * (z_max - z_min) / resolution]
        ray_direction = [1.0, 0.0, 0.0]  # along the x-axis
        ray_data.append((ray_origin, ray_direction))

# convert the list to a structured numpy array
rays = np.array(ray_data, dtype=Ray)


# Prepare an output array
output = np.zeros(len(rays), dtype=np.int32)
output_ = cuda.to_device(output)

# send rays and tris to dev
rays_ = cuda.to_device(rays)
tris_ = cuda.to_device(tris)
threads_per_block = 32
blocks_per_grid = (rays_.size + (threads_per_block - 1)) // threads_per_block

trace_rays[blocks_per_grid, threads_per_block](rays_, tris_, intersection_grid_, x_min, x_max, y_min, y_max, z_min, z_max, resolution)

intersection_grid_ = intersection_grid_.copy_to_host(intersection_grid)
print("Total intersections:", np.sum(intersection_grid))
