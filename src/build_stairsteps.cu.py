import numpy as np
from stl import mesh
from numba import cuda
from collections import defaultdict

from helpers import compute_bbox

ENABLE_CUDA = 1

if ENABLE_CUDA:
    from kernels import ray_intersects_tri, get_face_ids, trace_rays
else:
    from kernels_host import ray_intersects_tri, get_face_ids, trace_rays

THREADS_PER_BLOCK = 128

# load test STL
stl_mesh = mesh.Mesh.from_file('../utils/shapenet/2.stl')
# ensure triangles are numpy arrays for GPU usage
tris = np.array(stl_mesh.vectors, dtype=np.float32)#[7:8, :, :]
print(tris)
# keep tris on GPU while ray-tracing in all 3 directions
tris_ = cuda.to_device(tris)

# define ray data type; use structured numpy array for GPU usage
Ray = np.dtype([('origin',    np.float32, (3,)), 
                ('direction', np.float32, (3,))])

# create background structured mesh
x_min, x_max, y_min, y_max, z_min, z_max = compute_bbox(stl_mesh)
print("bounding box with buffer:", x_min, x_max, y_min, y_max, z_min, z_max)

resolution = 128

# coordinates of nodes
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
z = np.linspace(z_min, z_max, resolution)

# grid of nodes
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


def test_face_ids(xp, yp, zp):
    pass


def move_vertices_to_closest_node(stl_mesh, X, Y, Z):
    """
    Iterates over the vertices in the STL mesh and moves each vertex to the 
    closest node in the background mesh.

    """
    for i in range(len(stl_mesh.vectors)):
        for j in range(3):  # Each triangle has 3 vertices
            vertex = stl_mesh.vectors[i][j]
            # Calculate the differences for each dimension
            dx = np.abs(X - vertex[0])
            dy = np.abs(Y - vertex[1])
            dz = np.abs(Z - vertex[2])

            # Find the indices of the closest node
            closest_idx = np.unravel_index(np.argmin(dx**2 + dy**2 + dz**2), dx.shape)

            # Update the vertex to the closest node
            stl_mesh.vectors[i][j] = np.array([X[closest_idx], Y[closest_idx], Z[closest_idx]])

    return stl_mesh


stl_mesh.save('old.stl')

# shifted grid 
#dx = 0.0*(x[1]-x[0])
#dy = 0.0*(y[1]-y[0])
#dz = 0.0*(z[1]-z[0])

#xs = np.linspace(x_min+dx, x_max+dx, 23)
#ys = np.linspace(y_min+dy, y_max+dy, 23)
#zs = np.linspace(z_min+dz, z_max+dz, 23)

# grid of nodes 
#Xs, Ys, Zs = np.meshgrid(xs, ys, zs, indexing='ij')
#print(xs)

#stl_mesh = move_vertices_to_closest_node(stl_mesh, Xs, Ys, Zs)
#stl_mesh.save('new.stl')

def get_grid_index_host(point_x, point_y, point_z):
    """ 
    Given a point in the domain, return the index of the 
    CELL to which it belongs
    """
    x_index = int((point_x - x_min) / (x_max - x_min) * (resolution-1))
    y_index = int((point_y - y_min) / (y_max - y_min) * (resolution-1))
    z_index = int((point_z - z_min) / (z_max - z_min) * (resolution-1))
    # handle edge cases
    x_index = min(x_index, resolution - 2)
    y_index = min(y_index, resolution - 2)
    z_index = min(z_index, resolution - 2)

    return x_index, y_index, z_index


def get_face_nodes(face_index, dimension):
    """
    Retrieve coordinates of the four nodes forming a specific face.
    
    Given a face index on a grid, return the coordinates of the four nodes
    that make up the face. The face is determined by the 'dimension' (x, y, or z).
    The index refers to the face index in the specified dimension.
    """
    xid, yid, zid = face_index
    
    if dimension == 'x':
        nodes = [
            (X[xid, yid, zid],     Y[xid, yid, zid],     Z[xid, yid, zid]),
            (X[xid, yid+1, zid],   Y[xid, yid+1, zid],   Z[xid, yid+1, zid]),
            (X[xid, yid+1, zid+1], Y[xid, yid+1, zid+1], Z[xid, yid+1, zid+1]),
            (X[xid, yid, zid+1],   Y[xid, yid, zid+1],   Z[xid, yid, zid+1])
        ]
    elif dimension == 'y':
        nodes = [
            (X[xid, yid, zid],     Y[xid, yid, zid],     Z[xid, yid, zid]),
            (X[xid+1, yid, zid],   Y[xid+1, yid, zid],   Z[xid+1, yid, zid]),
            (X[xid+1, yid, zid+1], Y[xid+1, yid, zid+1], Z[xid+1, yid, zid+1]),
            (X[xid, yid, zid+1],   Y[xid, yid, zid+1],   Z[xid, yid, zid+1])
        ]
    elif dimension == 'z':
        nodes = [
            (X[xid, yid, zid],     Y[xid, yid, zid],     Z[xid, yid, zid]),
            (X[xid+1, yid, zid],   Y[xid+1, yid, zid],   Z[xid+1, yid, zid]),
            (X[xid+1, yid+1, zid], Y[xid+1, yid+1, zid], Z[xid+1, yid+1, zid]),
            (X[xid, yid+1, zid],   Y[xid, yid+1, zid],   Z[xid, yid+1, zid])
        ]

    return nodes


def create_tris_from_nodes(nodes, flag, dimension):
    """
    Creates two triangles from a set of four nodes.

    Take in four nodes (corners of a cell face), and create two triangles by
    splitting the rectangle along a diagonal. Return truangles formatted to 
    later save as STL (vertices in CCW order as viewed from outside the 
    surface). The `outside`ness is defined by the flag.

    """
    # first triangle uses the first diagonal
    triangle1 = [nodes[0], nodes[1], nodes[3]]

    # second triangle uses the same diagonal
    triangle2 = [nodes[1], nodes[2], nodes[3]]

    # orient triangles
    if flag == 'minus':
        # for the 'minus' flag, the normal should point in the -ve axis dir
        # inverting the order of the triangles will reverse the normal
        triangle1.reverse()
        triangle2.reverse()

    return triangle1, triangle2


def initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max, sample=True):
    """Generate a regular grid of rays along the specified axis, originating
       at the minimum value of the axis. The rays are centered in the face
       normal to the axis. If sample is True, generate staggered rays.
       NOTE: Looping only works for a cubic grid.
    """
    ray_data = []
    offset_factor = 0.01*(x[1]-x[0])  # Adjust this factor to control the offset

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

            # Add staggered rays if sampling is enabled
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

            # Add the base ray
            ray_data.append((base_origin, directions[0]))

    return np.array(ray_data, dtype=[('origin', 'f4', (3,)), ('direction', 'f4', (3,))])


def dump_rays_to_file(rays, filename):
    with open(filename, 'w') as file:
        for ray in rays:
            ray_origin = ray['origin']
            ray_direction = ray['direction']
            file.write(f"Origin: {ray_origin}, Direction: {ray_direction}\n")



def trace_host(axis):
    # intersects are cell-based, not node-based
    #intersects = np.zeros((resolution-1, resolution-1, resolution-1), dtype=np.bool_)
    if axis==0:
        intersects = np.zeros((resolution, resolution-1, resolution-1), dtype=np.bool_)
    elif axis==1:
        intersects = np.zeros((resolution-1, resolution, resolution-1), dtype=np.bool_)
    elif axis==2:
        intersects = np.zeros((resolution-1, resolution-1, resolution), dtype=np.bool_)
    rays = initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max)
    dump_rays_to_file(rays, f'rays{axis}.txt')

    trace_rays(rays, tris, intersects, x_min, x_max, y_min, y_max, z_min, z_max, resolution, axis)
    
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))
    return intersects


def trace(axis):
    # intersects are cell-based, not node-based
    #intersects = np.zeros((resolution-1, resolution-1, resolution-1), dtype=np.bool_)
    if axis==0:
        intersects = np.zeros((resolution, resolution-1, resolution-1), dtype=np.bool_)
    elif axis==1:
        intersects = np.zeros((resolution-1, resolution, resolution-1), dtype=np.bool_)
    elif axis==2:
        intersects = np.zeros((resolution-1, resolution-1, resolution), dtype=np.bool_)
    rays = initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max)
    dump_rays_to_file(rays, f'rays{axis}.txt')
    rays_ = cuda.to_device(rays)
    intersects_ = cuda.to_device(intersects)

    blocks_per_grid = max((rays.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 320)
    # STAGMOD
    intersection_flags = cuda.device_array(rays.shape[0], dtype=np.int32)
    trace_rays[blocks_per_grid, THREADS_PER_BLOCK] \
        (rays_, tris_, intersects_, x_min, x_max, y_min, y_max, z_min, z_max, resolution,\
        axis, intersection_flags)
    cuda.synchronize()
    
    intersects = intersects_.copy_to_host()
    del rays_, intersects_
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))
    # STAGMOD
    total_intersections = intersection_flags.copy_to_host().sum()
    print("Total violations:", total_intersections)
    return intersects


# perform ray tracing along each axis
if ENABLE_CUDA:
    x_intersects = trace(0)
    print(x_intersects.shape)
    y_intersects = trace(1)
    print(y_intersects.shape)
    z_intersects = trace(2)
    print(z_intersects.shape)
else:
    x_intersects = trace_host(0)
    print(x_intersects.shape)
    y_intersects = trace_host(1)
    print(y_intersects.shape)
    z_intersects = trace_host(2)
    print(z_intersects.shape)


# reconstruct stair-stepped mesh
all_triangles = []

# process x_intersects
for z_index in range(x_intersects.shape[2]):
    for y_index in range(x_intersects.shape[1]):
        flag = 'minus'  # always minus 
        for x_index in range(x_intersects.shape[0]):
            if x_intersects[x_index, y_index, z_index]:
                # get the nodes for the face
                nodes = get_face_nodes([x_index, y_index, z_index], 'x')
                # get the triangles from the nodes
                triangles = create_tris_from_nodes(nodes, flag, 'x')
                all_triangles.extend(triangles)

# process y_intersects
for z_index in range(y_intersects.shape[2]):
    for x_index in range(y_intersects.shape[0]):
        flag = 'minus'
        for y_index in range(y_intersects.shape[1]):
            if y_intersects[x_index, y_index, z_index]:
                nodes = get_face_nodes([x_index, y_index, z_index], 'y')
                triangles = create_tris_from_nodes(nodes, flag, 'y')
                all_triangles.extend(triangles)

# process z_intersects
for y_index in range(z_intersects.shape[1]):
    for x_index in range(z_intersects.shape[0]):
        flag = 'minus'
        for z_index in range(z_intersects.shape[2]):
            if z_intersects[x_index, y_index, z_index]:
                nodes = get_face_nodes([x_index, y_index, z_index], 'z')
                triangles = create_tris_from_nodes(nodes, flag, 'z')
                all_triangles.extend(triangles)


def remove_open_edges(triangles):
    edge_to_triangle = defaultdict(list)
    for i, triangle in enumerate(triangles):
        # Construct edges with sorted vertex indices to be comparable
        edges = [tuple(sorted([triangle[j], triangle[(j + 1) % 3]])) for j in range(3)]
        for edge in edges:
            edge_to_triangle[edge].append(i)

    # Identify edges that belong to only one triangle, these are open edges
    open_edges = {edge for edge, tris in edge_to_triangle.items() if len(tris) == 1}

    # Triangles to delete are those that have an open edge
    triangles_to_delete = set()
    for edge in open_edges:
        triangles_to_delete.update(edge_to_triangle[edge])

    # Return new list without the triangles that have open edges
    return [triangle for i, triangle in enumerate(triangles) if i not in triangles_to_delete]


# iteratively remove open edges until converged
previous_len = -1
current_len = len(all_triangles)

while previous_len != current_len:
    previous_len = current_len
    all_triangles = remove_open_edges(all_triangles)
    current_len = len(all_triangles)
    print(current_len)


# create mesh
num_triangles = len(all_triangles)
stl_mesh = mesh.Mesh(np.zeros(num_triangles, dtype=mesh.Mesh.dtype))

# fill the mesh with triangles
for i, triangle in enumerate(all_triangles):
    for j in range(3):
        stl_mesh.vectors[i][j] = triangle[j]

stl_mesh.save('output.stl')
