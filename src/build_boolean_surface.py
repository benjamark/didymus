import numpy as np
from stl import mesh
from numba import cuda
import trimesh
import os
from collections import defaultdict
from helpers import compute_bbox, center_bbox, dump_rays_to_file


ENABLE_CUDA = 1
BBOX_TYPE = 'manual'  # 'prop' or 'cube' or 'manual'
REMOVE_OPEN_EDGES = 0
ENABLE_RAY_SAMPLING = 1

if ENABLE_CUDA:
    from kernels import ray_intersects_tri, get_face_ids, trace_rays
else:
    from kernels_host import ray_intersects_tri, get_face_ids, trace_rays

path_to_bools = 'npys'
os.makedirs(path_to_bools, exist_ok=True)

# define ray data type; use structured numpy array for GPU usage
Ray = np.dtype([('origin',    np.float32, (3,)), 
                ('direction', np.float32, (3,))])


def initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, \
                    z_max, sample=False):
    """Generate a regular grid of rays along the specified axis, originating
       at the minimum value of the axis. The rays are centered in the face
       normal to the axis. If sample is True, generate staggered rays.
       NOTE: Looping only works for a cubic grid.
    """
    ray_data = []
    offset_factor = 0.01*(x[1]-x[0])  # offset from central ray

    for i in range(resolution-1):
        for j in range(resolution-1):
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
    # intersects are cell-based, not node-based
    if axis==0:
        intersects = np.zeros((resolution, resolution-1, resolution-1), \
                              dtype=np.bool_)
    elif axis==1:
        intersects = np.zeros((resolution-1, resolution, resolution-1), \
                              dtype=np.bool_)
    elif axis==2:
        intersects = np.zeros((resolution-1, resolution-1, resolution), \
                              dtype=np.bool_)

    rays = initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, \
                           z_min, z_max, sample=ENABLE_RAY_SAMPLING)

    trace_rays(rays, tris, intersects, x_min, x_max, y_min, y_max, z_min, \
               z_max, resolution, axis)
    
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))
    return intersects


def trace(axis, cell):
    if cell:
        intersects = np.zeros((resolution-1, resolution-1, resolution-1), \
                              dtype=np.bool_)
    else:
        # intersects are cell-based, not node-based
        if axis==0:
            intersects = np.zeros((resolution, resolution-1, resolution-1), \
                                  dtype=np.bool_)
        elif axis==1:
            intersects = np.zeros((resolution-1, resolution, resolution-1), \
                                  dtype=np.bool_)
        elif axis==2:
            intersects = np.zeros((resolution-1, resolution-1, resolution), \
                                  dtype=np.bool_)

    rays = initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, \
                           z_min, z_max, sample=ENABLE_RAY_SAMPLING)

    rays_ = cuda.to_device(rays)
    intersects_ = cuda.to_device(intersects)

    blocks_per_grid = max((rays.size + THREADS_PER_BLOCK - 1) \
                          // THREADS_PER_BLOCK, 320)
    # STAGMOD
    intersection_flags = cuda.device_array(rays.shape[0], dtype=np.int32)
    trace_rays[blocks_per_grid, THREADS_PER_BLOCK] \
        (rays_, tris_, intersects_, x_min, x_max, y_min, y_max, z_min, \
        z_max, resolution, axis, intersection_flags)

    cuda.synchronize()
    
    intersects = intersects_.copy_to_host()
    del rays_, intersects_
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))
    # STAGMOD
    total_intersections = intersection_flags.copy_to_host().sum()
    return intersects

corner_stls = ['../utils/shapenet/1.stl', '../utils/shapenet/2.stl']
simplex_dim = len(corner_stls)

# TODO:
# 1. implement loop over no. of corner cases. [OK]
# 2. automate bounding box and domain selection [OK]
# 3. remove `intersection_flags` and other STAGMODs
# 4. integrate host and device kernels
# 5. combine three `intersects` arrays into one for GPU
# 6. implement block ray processing
# 7. add buffer to bbox [OK]
# 8. refactor helpers to helpers

THREADS_PER_BLOCK = 128

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

resolution = 32

# coordinates of nodes
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
z = np.linspace(z_min, z_max, resolution)
# grid of nodes
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

for idx, stl_file in enumerate(corner_stls):
    stl_mesh = mesh.Mesh.from_file(stl_file)
    stl_mesh = center_bbox(stl_mesh)

    # ensure triangles are numpy arrays for GPU usage
    tris = np.array(stl_mesh.vectors, dtype=np.float32)#[7:8, :, :]
    # keep tris on GPU while ray-tracing in all 3 directions
    tris_ = cuda.to_device(tris)

    # perform ray tracing along each axis
    if ENABLE_CUDA:
        print(f'Ray-tracing on device with {resolution*resolution} rays')
        x_intersects = trace(0, cell='True')
        y_intersects = trace(1, cell='True')
        z_intersects = trace(2, cell='True')
    else:
        print(f'Ray-tracing on host with {resolution*resolution} rays')
        x_intersects = trace_host(0)
        y_intersects = trace_host(1)
        z_intersects = trace_host(2)

    print(np.sum(x_intersects))
    print(np.sum(y_intersects))
    print(np.sum(z_intersects))
    intersects = x_intersects +y_intersects +z_intersects
    print(np.sum(intersects))
    np.save(f'{path_to_bools}/corner_{idx}.npy', intersects)

