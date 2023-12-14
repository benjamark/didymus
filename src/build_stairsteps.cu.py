import numpy as np
from stl import mesh
from numba import cuda
from kernels import ray_intersects_tri, get_grid_index, trace_rays

THREADS_PER_BLOCK = 128  

# load test STL
sphere_mesh = mesh.Mesh.from_file('../utils/sphere.stl')
# ensure triangles are numpy arrays for GPU usage
tris = np.array(sphere_mesh.vectors, dtype=np.float32)
# keep tris on GPU while ray-tracing in all 3 directions
tris_ = cuda.to_device(tris)

# define ray data type; use structured numpy array for GPU usage
Ray = np.dtype([('origin',    np.float32, (3,)), 
                ('direction', np.float32, (3,))])

# create background structured mesh
x_min, x_max = -1.2, 1.2
y_min, y_max = -1.2, 1.2
z_min, z_max = -1.2, 1.2
resolution = 512

def initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max):
    """Generate a regular grid of rays along the specified axis, originating
       at the minimum value of the axis. The rays are centered in the face
       normal to the axis.
    """
    ray_data = []
    for i in range(resolution):
        for j in range(resolution):
            if axis == 'x':
                ray_origin = [x_min, y_min + (i + 0.5) / resolution * (y_max - y_min), z_min + (j + 0.5) / resolution * (z_max - z_min)]
                ray_direction = [1.0, 0.0, 0.0]
            elif axis == 'y':
                ray_origin = [x_min + (i + 0.5) / resolution * (x_max - x_min), y_min, z_min + (j + 0.5) / resolution * (z_max - z_min)]
                ray_direction = [0.0, 1.0, 0.0]
            else:  # axis == 'z'
                ray_origin = [x_min + (i + 0.5) / resolution * (x_max - x_min), y_min + (j + 0.5) / resolution * (y_max - y_min), z_min]
                ray_direction = [0.0, 0.0, 1.0]
            ray_data.append((ray_origin, ray_direction))
    return np.array(ray_data, dtype=Ray)

def dump_rays_to_file(rays, filename):
    with open(filename, 'w') as file:
        for ray in rays:
            ray_origin = ray['origin']
            ray_direction = ray['direction']
            file.write(f"Origin: {ray_origin}, Direction: {ray_direction}\n")


def trace(axis):
    intersects = np.zeros((resolution, resolution, resolution), dtype=np.bool_)
    rays = initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max)
    rays_ = cuda.to_device(rays)
    intersects_ = cuda.to_device(intersects)

    blocks_per_grid = max((rays.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 320)
    trace_rays[blocks_per_grid, THREADS_PER_BLOCK](rays_, tris_, intersects_, x_min, x_max, y_min, y_max, z_min, z_max, resolution)
    cuda.synchronize()
    
    intersects = intersects_.copy_to_host()
    del rays_, intersects_
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))
    return intersects

# Perform ray tracing along each axis
x_intersects = trace('x')
y_intersects = trace('y')
z_intersects = trace('z')
