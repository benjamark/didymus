import numpy as np
from stl import mesh
from numba import cuda
import trimesh
import os
from collections import defaultdict
from helpers import compute_bbox, center_bbox, dump_rays_to_file, load_config
import time

# load configuration file
config = load_config()

corner_stls = config["corner_stls"]
ENABLE_CUDA = config["ENABLE_CUDA"]
ENABLE_RAY_SAMPLING = config["ENABLE_RAY_SAMPLING"]
THREADS_PER_BLOCK = config["THREADS_PER_BLOCK"]
resolution = config["resolution"]
project_dir = config["project_dir"]


if ENABLE_CUDA:
    from kernels import ray_intersects_tri, get_cell_ids, trace_rays
else:
    from kernels_host import ray_intersects_tri, get_cell_ids, trace_rays

os.makedirs(project_dir, exist_ok=True)

# define ray data type; use structured numpy array for GPU usage
Ray = np.dtype([('origin',    np.float32, (3,)), 
                ('direction', np.float32, (3,))])


def initialize_rays(axis, sample=False):
    """Generate a regular grid of rays along the specified axis, originating
       at the minimum value of the axis. The rays are centered in the face
       normal to the axis. If sample is True, generate staggered rays.
       NOTE: Looping only works for a cubic grid.
    """
    ray_data = []
    offset_factor = 0.01*(x[1]-x[0])  # offset from central ray

    if axis==0:
        res_i = resolution_y
        res_j = resolution_z
    elif axis==1:
        res_i = resolution_x
        res_j = resolution_z
    elif axis==2:
        res_i = resolution_x
        res_j = resolution_y

    for i in range(res_i-1):
        for j in range(res_j-1):
            # base ray origin calculation
            if axis==0:
                base_origin = [x_min, 0.5*(y[i+1]+y[i]), 0.5*(z[j+1]+z[j])]
                directions = [(1.0, 0.0, 0.0)]
            elif axis==1:
                base_origin = [0.5*(x[i+1]+x[i]), y_min, 0.5*(z[j+1]+z[j])]
                directions = [(0.0, 1.0, 0.0)]
            else:  # axis == 2
                base_origin = [0.5*(x[i+1]+x[i]), 0.5*(y[j+1]+y[j]), z_min]
                directions = [(0.0, 0.0, 1.0)]

            # add staggered rays if sampling is enabled
            if sample:
                if axis==0:
                    offsets = [(0, offset_factor, 0), (0, -offset_factor, 0), 
                               (0, 0, offset_factor), (0, 0, -offset_factor)]
                elif axis==1:
                    offsets = [(offset_factor, 0, 0), (-offset_factor, 0, 0), 
                               (0, 0, offset_factor), (0, 0, -offset_factor)]
                else: 
                    offsets = [(offset_factor, 0, 0), (-offset_factor, 0, 0), 
                               (0, offset_factor, 0), (0, -offset_factor, 0)]

                for offset in offsets:
                    staggered_origin = [base_origin[0] + offset[0], 
                                        base_origin[1] + offset[1], 
                                        base_origin[2] + offset[2]]
                    ray_data.append((staggered_origin, directions[0]))

            # add the base ray
            ray_data.append((base_origin, directions[0]))

    return np.array(ray_data, dtype=[('origin', 'f4', (3,)), \
                                     ('direction', 'f4', (3,))])


def trace_host(axis):
    # TODO: make cell mandatory
    intersects = np.zeros((resolution_x-1, resolution_y-1, resolution_z-1), \
                          dtype=np.bool_)

    rays = initialize_rays(axis, sample=ENABLE_RAY_SAMPLING)

    trace_rays(rays, tris, intersects, x_min, x_max, y_min, y_max, z_min, \
               z_max, resolution_x, resolution_y, resolution_z, axis)
    
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))
    return intersects


def trace(axis):
    intersects = np.zeros((resolution_x-1, resolution_y-1, resolution_z-1), \
                          dtype=np.bool_)

    rays = initialize_rays(axis, sample=ENABLE_RAY_SAMPLING)

    rays_ = cuda.to_device(rays)
    intersects_ = cuda.to_device(intersects)

    blocks_per_grid = max((rays.size + THREADS_PER_BLOCK - 1) \
                          // THREADS_PER_BLOCK, 320)

    trace_rays[blocks_per_grid, THREADS_PER_BLOCK] \
        (rays_, tris_, intersects_, x_min, x_max, y_min, y_max, z_min, \
        z_max, resolution_x, resolution_y, resolution_z, axis)

    cuda.synchronize()
    
    intersects = intersects_.copy_to_host()
    del rays_, intersects_
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))

    return intersects

# initialize domain boundaries
x_min, x_max = np.inf, -np.inf
y_min, y_max = np.inf, -np.inf
z_min, z_max = np.inf, -np.inf

# compute domain boundaries based on all corner stls
for stl_file in corner_stls:
    stl_mesh = mesh.Mesh.from_file(stl_file)
    stl_mesh = center_bbox(stl_mesh)
    minx, maxx, miny, maxy, minz, maxz = compute_bbox(stl_mesh)

    # update global bounding box
    x_min = min(x_min, minx)
    x_max = max(x_max, maxx)
    y_min = min(y_min, miny)
    y_max = max(y_max, maxy)
    z_min = min(z_min, minz)
    z_max = max(z_max, maxz)

print("bounding box with buffer:", x_min, x_max, y_min, y_max, z_min, z_max)

x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

# identify the largest dimension range and calculate dx
largest_range = max(x_range, y_range, z_range)
dx = largest_range / resolution

# adjust dimensions to nearest integral multiple of dx
if largest_range == x_range:
    y_max = y_min + round(y_range / dx) * dx
    z_max = z_min + round(z_range / dx) * dx
    resolution_x = resolution
    resolution_y = round((y_max - y_min) / dx)
    resolution_z = round((z_max - z_min) / dx)
elif largest_range == y_range:
    x_max = x_min + round(x_range / dx) * dx
    z_max = z_min + round(z_range / dx) * dx
    resolution_x = round((x_max - x_min) / dx)
    resolution_y = resolution
    resolution_z = round((z_max - z_min) / dx)
else:  # z_range is the largest
    x_max = x_min + round(x_range / dx) * dx
    y_max = y_min + round(y_range / dx) * dx
    resolution_x = round((x_max - x_min) / dx)
    resolution_y = round((y_max - y_min) / dx)
    resolution_z = resolution

print("adjusted bounding box:", x_min, x_max, y_min, y_max, z_min, z_max)

# generate grid 
x = np.linspace(x_min, x_max, resolution_x)
y = np.linspace(y_min, y_max, resolution_y)
z = np.linspace(z_min, z_max, resolution_z)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


for idx, stl_file in enumerate(corner_stls):
    stl_mesh = mesh.Mesh.from_file(stl_file)
    stl_mesh = center_bbox(stl_mesh)

    # ensure triangles are numpy arrays for GPU usage
    tris = np.array(stl_mesh.vectors, dtype=np.float32)
    # keep tris on GPU while ray-tracing in all 3 directions
    tris_ = cuda.to_device(tris)

    # perform ray tracing along each axis
    if ENABLE_CUDA:
        t0 = time.time()
        print(f'Ray-tracing on device with {resolution_y*resolution_z} rays')
        x_intersects = trace(0)
        print(f'Ray-tracing on device with {resolution_y*resolution_x} rays')
        y_intersects = trace(1)
        print(f'Ray-tracing on device with {resolution_x*resolution_y} rays')
        z_intersects = trace(2)
        t1 = time.time()
        print(f'Timings :: ray-tracing: {t1-t0} (s)')
    else:
        t0 = time.time()
        print(f'Ray-tracing on host with {resolution_y*resolution_z} rays')
        x_intersects = trace_host(0)
        print(f'Ray-tracing on host with {resolution_y*resolution_x} rays')
        y_intersects = trace_host(1)
        print(f'Ray-tracing on host with {resolution_x*resolution_y} rays')
        print(f'Timings :: ray-tracing: {t1-t0} (s)')
        t1 = time.time()
        z_intersects = trace_host(2)

    intersects = x_intersects +y_intersects +z_intersects
    print(f'Total no. of intersects:{np.sum(intersects)}')
    print(f'Total no. of grid cells:{resolution_x*resolution_y*resolution_z}')
    print(f'Fraction of filled volume:{np.sum(intersects)/resolution**3}')
    np.save(f'{project_dir}/corner_{idx}.npy', intersects)
