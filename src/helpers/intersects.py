import numpy as np

def ray_intersects_triangle(ray, triangle):
    """ Moller-Trumbore algorithm to detect ray intersection
        with a single triangle and return the intersection point.
    """
    # get triangle vertices
    v1, v2, v3 = triangle

    # Vectors for two edges sharing v1
    edge1 = v2 - v1
    edge2 = v3 - v1

    # calculate determinant
    pvec = np.cross(ray.direction, edge2)
    det = edge1.dot(pvec)

    # if determinant is near zero, ray lies in plane of triangle
    if abs(det) < 1e-8:
        return False, None

    inv_det = 1.0 / det

    # calculate distance from v1 to ray origin
    tvec = ray.origin - v1

    # calculate U parameter and test bounds
    u = tvec.dot(pvec) * inv_det
    if u < 0 or u > 1:
        return False, None

    # prepare to test V parameter
    qvec = np.cross(tvec, edge1)

    # calculate V parameter and test bounds
    v = ray.direction.dot(qvec) * inv_det
    if v < 0 or u + v > 1:
        return False, None

    # calculate t, the distance from v1 to intersection point
    t = edge2.dot(qvec) * inv_det

    # calculate intersection point
    intersection_point = ray.origin + t * ray.direction

    return True, intersection_point

