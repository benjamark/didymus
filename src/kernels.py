from numba import cuda
import numpy as np

@cuda.jit(device=True)
def ray_intersects_tri(ray, triangle):
    # Get triangle vertices
    v1, v2, v3 = triangle

    # Manually calculate vectors for two edges sharing v1
    edge1_x = v2[0] - v1[0]
    edge1_y = v2[1] - v1[1]
    edge1_z = v2[2] - v1[2]

    edge2_x = v3[0] - v1[0]
    edge2_y = v3[1] - v1[1]
    edge2_z = v3[2] - v1[2]

    # Calculate cross product of ray direction and edge2 (pvec)
    pvec_x = ray['direction'][1] * edge2_z - ray['direction'][2] * edge2_y
    pvec_y = ray['direction'][2] * edge2_x - ray['direction'][0] * edge2_z
    pvec_z = ray['direction'][0] * edge2_y - ray['direction'][1] * edge2_x

    # Dot product of edge1 and pvec
    det = edge1_x * pvec_x + edge1_y * pvec_y + edge1_z * pvec_z

    # If determinant is near zero, ray lies in plane of triangle
    if abs(det) < 1e-8:
        return False, None, None, None

    inv_det = 1.0 / det

    # Calculate distance from v1 to ray origin
    tvec_x = ray['origin'][0] - v1[0]
    tvec_y = ray['origin'][1] - v1[1]
    tvec_z = ray['origin'][2] - v1[2]

    # Calculate u parameter and test bounds
    u = (tvec_x * pvec_x + tvec_y * pvec_y + tvec_z * pvec_z) * inv_det
    if u < 0 or u > 1:
        return False, None, None, None

    # Prepare to test v parameter
    qvec_x = tvec_y * edge1_z - tvec_z * edge1_y
    qvec_y = tvec_z * edge1_x - tvec_x * edge1_z
    qvec_z = tvec_x * edge1_y - tvec_y * edge1_x

    # Calculate v parameter and test bounds
    v = (ray['direction'][0] * qvec_x \
       + ray['direction'][1] * qvec_y \
       + ray['direction'][2] * qvec_z) * inv_det
    if v < 0 or u + v > 1:
        return False, None, None, None

    # Calculate t, the distance from v1 to the intersection point
    t = (edge2_x * qvec_x + edge2_y * qvec_y + edge2_z * qvec_z) * inv_det

    # Calculate intersection point
    intersection_point_x = ray['origin'][0] + t * ray['direction'][0]
    intersection_point_y = ray['origin'][1] + t * ray['direction'][1]
    intersection_point_z = ray['origin'][2] + t * ray['direction'][2]

    return True, intersection_point_x, intersection_point_y, intersection_point_z


#@cuda.jit(device=True)
#def get_grid_index(point_x, point_y, point_z, x_min, x_max, y_min, y_max, z_min, z_max, resolution):
#    """ 
#    Given a point in the domain, return the index of the 
#    CELL to which it belongs
#
#    """
#    x_index = int((point_x - x_min) / (x_max - x_min) * (resolution-1))
#    y_index = int((point_y - y_min) / (y_max - y_min) * (resolution-1))
#    z_index = int((point_z - z_min) / (z_max - z_min) * (resolution-1))
#    # handle edge cases
#    x_index = min(x_index, resolution - 2)
#    y_index = min(y_index, resolution - 2)
#    z_index = min(z_index, resolution - 2)
#    return x_index, y_index, z_index


@cuda.jit(device=True)
def get_face_ids(point_x, point_y, point_z, x_min, x_max, y_min, y_max, z_min, z_max, resolution, axis):
    """
    Given a point in the domain, return the ID of the nearest FACE in the grid
    for each dimension (x, y, z). The nearest face is determined by checking if the
    point is closer to the 'min' or 'max' face in each dimension.
    """
    if axis ==0:
        # normalize the point coordinates within the grid range
        x_normalized = (point_x - x_min) / (x_max - x_min)
        xf_id = round(x_normalized*(resolution-1))

        y_normalized = (point_y - y_min) / (y_max - y_min)
        yf_id = round(x_normalized*(resolution-2))

        z_normalized = (point_z - z_min) / (z_max - z_min)
        zf_id = round(x_normalized*(resolution-2))

        return xf_id, yf_id, zf_id

    elif axis ==1:
        # normalize the point coordinates within the grid range
        x_normalized = (point_x - x_min) / (x_max - x_min)
        xf_id = round(x_normalized*(resolution-2))

        y_normalized = (point_y - y_min) / (y_max - y_min)
        yf_id = round(x_normalized*(resolution-1))

        z_normalized = (point_z - z_min) / (z_max - z_min)
        zf_id = round(x_normalized*(resolution-2))

        return xf_id, yf_id, zf_id

    elif axis ==2:
        # normalize the point coordinates within the grid range
        x_normalized = (point_x - x_min) / (x_max - x_min)
        xf_id = round(x_normalized*(resolution-2))

        y_normalized = (point_y - y_min) / (y_max - y_min)
        yf_id = round(x_normalized*(resolution-2))

        z_normalized = (point_z - z_min) / (z_max - z_min)
        zf_id = round(x_normalized*(resolution-1))

        return xf_id, yf_id, zf_id

@cuda.jit
def trace_rays(rays, triangles, intersection_grid, x_min, x_max, y_min, y_max, z_min, z_max, resolution, axis):
    ray_idx = cuda.grid(1)
    if ray_idx < rays.shape[0]:
        ray = rays[ray_idx]
        for j in range(triangles.shape[0]):
            intersects, point_x, point_y, point_z = ray_intersects_tri(ray, triangles[j])
            if intersects:
                x_idx, y_idx, z_idx = get_face_ids( \
                point_x, point_y, point_z, x_min, x_max, y_min, y_max, z_min, z_max, resolution, axis)
                intersection_grid[x_idx, y_idx, z_idx] = True
