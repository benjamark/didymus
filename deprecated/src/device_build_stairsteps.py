import numpy as np
from numba import cuda
from stl import mesh
from helpers.ray_tracing import ray_intersects_triangle

Ray = np.dtype([('origin', np.float32, (3,)), ('direction', np.float32, (3,))])

@cuda.jit(device=True)
def get_cell_index(point, x_min, x_max, y_min, y_max, z_min, z_max, resolution):
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
intersection_grid = np.zeros((resolution, resolution, resolution), dtype=bool)

# Initialize an empty list to store the ray data
ray_data = []
for y in range(resolution):
    for z in range(resolution):
        ray_origin = [x_min, y_min + (y + 0.5) * (y_max - y_min) / resolution, z_min + (z + 0.5) * (z_max - z_min) / resolution]
        ray_direction = [1.0, 0.0, 0.0]  # along the x-axis
        ray_data.append((ray_origin, ray_direction))

# Convert the list to a structured numpy array
rays = np.array(ray_data, dtype=Ray)


@cuda.jit
def ray_tracing_kernel(rays, triangles, grid, x_min, x_max, y_min, y_max, z_min, z_max, resolution):
    i = cuda.grid(1)  # Unique thread index

    if i < rays.shape[0]:
        ray = rays[i]
        first_intersection = None
        last_intersection = None

        for j in range(triangles.shape[0]):
            intersects, point = ray_intersects_triangle(ray, triangles[j])
            if intersects:
                if first_intersection is None:
                    first_intersection = point
                last_intersection = point

        if first_intersection is not None:
            x_index, y_index, z_index = get_cell_index(first_intersection, x_min, x_max, y_min, y_max, z_min, z_max, resolution)
            grid[x_index, y_index, z_index] = True

        if last_intersection is not None:
            x_index, y_index, z_index = get_cell_index(last_intersection, x_min, x_max, y_min, y_max, z_min, z_max, resolution)
            grid[x_index, y_index, z_index] = True

# Prepare data for GPU
rays_device = cuda.to_device(np.array(rays))
triangles_device = cuda.to_device(np.ascontiguousarray(stl.vectors))
grid_device = cuda.to_device(intersection_grid)

# Launch the kernel
threads_per_block = 128  # This is a typical value; adjust based on your GPU's capability
blocks_per_grid = (rays_device.size + (threads_per_block - 1)) // threads_per_block
ray_tracing_kernel[blocks_per_grid, threads_per_block](rays_device, triangles_device, grid_device, x_min, x_max, y_min, y_max, z_min, z_max, resolution)

# Copy the grid back to the host
grid_device.copy_to_host(intersection_grid)
np.save('grid.npy', intersection_grid)

