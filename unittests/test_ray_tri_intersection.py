import numpy as np
import sys
sys.path.append('../src')  

from helpers.ray_tracing import Ray, ray_intersects_triangle


def test_ray_triangle_intersection():
    triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])

    ray1 = Ray([0.25, 0.25, 1.], [0., 0., -1.])

    # non-intersecting ray
    ray2 = Ray([2., 2., 2.], [0., 1., 0.])

    # test intersecting ray
    intersects1, point1 = ray_intersects_triangle(ray1, triangle)
    assert intersects1, "Ray should intersect the triangle"

    # test non-intersecting ray
    intersects2, point2 = ray_intersects_triangle(ray2, triangle)
    assert not intersects2, "Ray should not intersect the triangle"

    print("Test 'test_ray_triangle_intersection' passed.")

if __name__ == "__main__":
    test_ray_triangle_intersection()
