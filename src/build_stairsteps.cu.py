import numpy as np
from stl import mesh
from numba import cuda
from helpers import compute_bbox
from kernels import ray_intersects_tri, get_grid_index, trace_rays

THREADS_PER_BLOCK = 128  

# load test STL
stl_mesh = mesh.Mesh.from_file('../utils/sphere.stl')
# ensure triangles are numpy arrays for GPU usage
tris = np.array(stl_mesh.vectors, dtype=np.float32)
# keep tris on GPU while ray-tracing in all 3 directions
tris_ = cuda.to_device(tris)

# define ray data type; use structured numpy array for GPU usage
Ray = np.dtype([('origin',    np.float32, (3,)), 
                ('direction', np.float32, (3,))])

# create background structured mesh
x_min, x_max, y_min, y_max, z_min, z_max = compute_bbox(stl_mesh)
print("bounding box with buffer:", x_min, x_max, y_min, y_max, z_min, z_max)

resolution = 256

# coordinates of nodes
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
z = np.linspace(z_min, z_max, resolution)

# grid of nodes 
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# coordinates of centers
xc = (x[:-1] + x[1:]) / 2
yc = (y[:-1] + y[1:]) / 2
zc = (z[:-1] + z[1:]) / 2


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


def get_face_nodes(cell_index, flag, dimension):
    """
    Retrieve coordinates of the four nodes forming a specific face of a cell.

    Given a cell index in the grid, return the coordinates of the four nodes
    that make up a selected face of the cell. The face is determined by the 
    'dimension' (x, y, or z) and the 'flag' ('plus' or 'minus'). 
    'plus' indicates the face on the higher value side of the specified 
    dimension, while 'minus' indicates the lower value side.

    """
    x_index, y_index, z_index = cell_index

    if flag == 'plus':
        if dimension == 'x':
            x_index += 1
        elif dimension == 'y':
            y_index += 1
        elif dimension == 'z':
            z_index += 1
    elif flag == 'minus':
        pass
    else:
        raise ValueError("flag must be 'plus' or 'minus'")

    if dimension == 'x':
        nodes = [
            (X[x_index, y_index, z_index], Y[x_index, y_index, z_index], 
             Z[x_index, y_index, z_index]),
            (X[x_index, y_index+1, z_index], Y[x_index, y_index+1, z_index], 
             Z[x_index, y_index+1, z_index]),
            (X[x_index, y_index+1, z_index+1], Y[x_index, y_index+1, z_index+1], 
             Z[x_index, y_index+1, z_index+1]),
            (X[x_index, y_index, z_index+1], Y[x_index, y_index, z_index+1], 
             Z[x_index, y_index, z_index+1])
        ]
    elif dimension == 'y':
        nodes = [
            (X[x_index, y_index, z_index], Y[x_index, y_index, z_index], 
             Z[x_index, y_index, z_index]),
            (X[x_index+1, y_index, z_index], Y[x_index+1, y_index, z_index], 
             Z[x_index+1, y_index, z_index]),
            (X[x_index+1, y_index, z_index+1], Y[x_index+1, y_index, z_index+1], 
             Z[x_index+1, y_index, z_index+1]),
            (X[x_index, y_index, z_index+1], Y[x_index, y_index, z_index+1], 
             Z[x_index, y_index, z_index+1])
        ]
    elif dimension == 'z':
        nodes = [
            (X[x_index, y_index, z_index], Y[x_index, y_index, z_index], 
             Z[x_index, y_index, z_index]),
            (X[x_index+1, y_index, z_index], Y[x_index+1, y_index, z_index], 
             Z[x_index+1, y_index, z_index]),
            (X[x_index+1, y_index+1, z_index], Y[x_index+1, y_index+1, z_index], 
             Z[x_index+1, y_index+1, z_index]),
            (X[x_index, y_index+1, z_index], Y[x_index, y_index+1, z_index], 
             Z[x_index, y_index+1, z_index])
        ]

    if flag == 'minus':
        nodes = nodes[::-1]

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


#def initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max):
#    """Generate a regular grid of rays along the specified axis, originating
#       at the minimum value of the axis. The rays are centered in the face
#       normal to the axis.
#    """
#    ray_data = []
#    for i in range(resolution):
#        for j in range(resolution):
#            if axis == 'x':
#                ray_origin = [x_min, \
#                y_min + (i + 0.5) / resolution *(y_max - y_min), \
#                z_min + (j + 0.5) / resolution *(z_max - z_min)]
#                ray_direction = [1.0, 0.0, 0.0]
#            elif axis == 'y':
#                ray_origin = [x_min + (i + 0.5) / resolution *(x_max - x_min), \
#                y_min, \
#                z_min + (j + 0.5) / resolution *(z_max - z_min)]
#                ray_direction = [0.0, 1.0, 0.0]
#            else:  # axis == 'z'
#                ray_origin = [x_min + (i + 0.5) / resolution *(x_max - x_min), \
#                y_min + (j + 0.5) / resolution *(y_max - y_min), \
#                z_min]
#                ray_direction = [0.0, 0.0, 1.0]
#            ray_data.append((ray_origin, ray_direction))
#    return np.array(ray_data, dtype=Ray)



def initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max, sample=False):
    """Generate a regular grid of rays along the specified axis, originating
       at the minimum value of the axis. The rays are centered in the face
       normal to the axis. If sample is True, generate staggered rays.
    """
    ray_data = []
    offset_factor = 5e-3  # Adjust this factor to control the offset

    for i in range(resolution):
        for j in range(resolution):
            # Base ray origin calculation
            if axis == 'x':
                base_origin = [x_min, y_min + (i + 0.5) / resolution * (y_max - y_min), 
                               z_min + (j + 0.5) / resolution * (z_max - z_min)]
                directions = [(1.0, 0.0, 0.0)]
            elif axis == 'y':
                base_origin = [x_min + (i + 0.5) / resolution * (x_max - x_min), y_min, 
                               z_min + (j + 0.5) / resolution * (z_max - z_min)]
                directions = [(0.0, 1.0, 0.0)]
            else:  # axis == 'z'
                base_origin = [x_min + (i + 0.5) / resolution * (x_max - x_min), 
                               y_min + (j + 0.5) / resolution * (y_max - y_min), z_min]
                directions = [(0.0, 0.0, 1.0)]

            # Add staggered rays if sampling is enabled
            if sample:
                if axis == 'x':
                    offsets = [(0, offset_factor, 0), (0, -offset_factor, 0), 
                               (0, 0, offset_factor), (0, 0, -offset_factor)]
                elif axis == 'y':
                    offsets = [(offset_factor, 0, 0), (-offset_factor, 0, 0), 
                               (0, 0, offset_factor), (0, 0, -offset_factor)]
                else:  # axis == 'z'
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


def trace(axis):
    # intersects are cell-based, not node-based
    intersects = np.zeros((resolution-1, resolution-1, resolution-1), dtype=np.bool_)
    rays = initialize_rays(axis, resolution, x_min, x_max, y_min, y_max, z_min, z_max)
    rays_ = cuda.to_device(rays)
    intersects_ = cuda.to_device(intersects)

    blocks_per_grid = max((rays.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 320)
    trace_rays[blocks_per_grid, THREADS_PER_BLOCK] \
        (rays_, tris_, intersects_, x_min, x_max, y_min, y_max, z_min, z_max, resolution)
    cuda.synchronize()
    
    intersects = intersects_.copy_to_host()
    del rays_, intersects_
    print(f"Total intersections along {axis}-axis:", np.sum(intersects))
    return intersects

# perform ray tracing along each axis
x_intersects = trace('x')
y_intersects = trace('y')
z_intersects = trace('z')


# reconstruct stair-stepped mesh
all_triangles = []

# process x_intersects
for z_index in range(x_intersects.shape[2]):
    for y_index in range(x_intersects.shape[1]):
        flag = 'minus'  # start with 'minus' for each new line in x
        for x_index in range(x_intersects.shape[0]):
            if x_intersects[x_index, y_index, z_index]:
                # get the nodes for the face
                nodes = get_face_nodes([x_index, y_index, z_index], flag, 'x')
                # get the triangles from the nodes
                triangles = create_tris_from_nodes(nodes, flag, 'x')
                all_triangles.extend(triangles)
                # alternate the flag
                flag = 'plus' if flag == 'minus' else 'minus'

# process y_intersects
for z_index in range(y_intersects.shape[2]):
    for x_index in range(y_intersects.shape[0]):
        flag = 'minus'
        for y_index in range(y_intersects.shape[1]):
            if y_intersects[x_index, y_index, z_index]:
                nodes = get_face_nodes([x_index, y_index, z_index], flag, 'y')
                triangles = create_tris_from_nodes(nodes, flag, 'y')
                all_triangles.extend(triangles)
                flag = 'plus' if flag == 'minus' else 'minus'

# process z_intersects
for y_index in range(z_intersects.shape[1]):
    for x_index in range(z_intersects.shape[0]):
        flag = 'minus'
        for z_index in range(z_intersects.shape[2]):
            if z_intersects[x_index, y_index, z_index]:
                nodes = get_face_nodes([x_index, y_index, z_index], flag, 'z')
                triangles = create_tris_from_nodes(nodes, flag, 'z')
                all_triangles.extend(triangles)
                flag = 'plus' if flag == 'minus' else 'minus'

# create mesh
num_triangles = len(all_triangles)
stl_mesh = mesh.Mesh(np.zeros(num_triangles, dtype=mesh.Mesh.dtype))

# fill the mesh with triangles
for i, triangle in enumerate(all_triangles):
    for j in range(3):
        stl_mesh.vectors[i][j] = triangle[j]

stl_mesh.save('output.stl')
