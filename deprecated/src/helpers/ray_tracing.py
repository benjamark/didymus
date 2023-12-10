from numba import cuda
import numpy as np

@cuda.jit(device=True)
def ray_intersects_triangle(ray, triangle):
    # get triangle vertices
    v1, v2, v3 = triangle

    # Element-wise vector operations
    edge1 = np.array([v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]], dtype=np.float32)
    edge2 = np.array([v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]], dtype=np.float32)

    # Remaining calculations
    pvec = np.cross(ray['direction'], edge2)
    det = edge1.dot(pvec)

    if abs(det) < 1e-8:
        return False, None

    inv_det = 1.0 / det

    # Element-wise subtraction
    tvec = np.array([ray['origin'][0] - v1[0], ray['origin'][1] - v1[1], ray['origin'][2] - v1[2]], dtype=np.float32)

    u = tvec.dot(pvec) * inv_det
    if u < 0 or u > 1:
        return False, None

    qvec = np.cross(tvec, edge1)
    v = ray['direction'].dot(qvec) * inv_det
    if v < 0 or u + v > 1:
        return False, None

    t = edge2.dot(qvec) * inv_det

    # Element-wise addition for intersection point
    intersection_point = np.array([ray['origin'][0] + t * ray['direction'][0],
                                   ray['origin'][1] + t * ray['direction'][1],
                                   ray['origin'][2] + t * ray['direction'][2]], dtype=np.float32)

    return True, intersection_point
