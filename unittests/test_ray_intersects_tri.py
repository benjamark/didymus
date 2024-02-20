import pytest
import numpy as np
import sys
sys.path.append('../src')
from kernels_host import ray_intersects_tri

Ray = np.dtype([('origin', np.float32, (3,)), ('direction', np.float32, (3,))])

@pytest.fixture
def create_ray_and_tris():
    def _create(ray_params, tri_params):
        ray = np.array([ray_params], dtype=Ray)
        tris = np.array(tri_params, dtype=np.float32)
        return ray, tris
    return _create

@pytest.mark.parametrize(
    "ray_params, tri_params, expected_output, expected_intersection",
    [
        # Test 1
        (
            ([0, 0, 0], [0, 1, 0]),  # ray
            [([1, 1, 0], [1, -1, 0], [1, 0, 1]), \
            ([-1, 0, 0], [1, 0, 0], [0, 1, 1])],  # tris
            [False, True],  # expected_output
            None  # expected_intersection
        ),
        # Test 2
        (
            ([0, 0, -1], [0, 0, 1]), 
            [([1, 1, 0], [1, -1, 0], [-1, -1, 0])],  
            [True],  
            [[0, 0, 0]] 
        ),
        # Test 3
        (
            ([0, 0, 1], [1, 0, 0]),  
            [([1, 1, 0], [1, -1, 0], [-1, 0, 0])],  
            [False],  
            None 
        ),
    ]
)


def test_ray_tri_intersection(create_ray_and_tris, ray_params, tri_params, \
                              expected_output, expected_intersection):
    ray, tris = create_ray_and_tris(ray_params, tri_params)
    output = np.zeros((1, len(tris)), dtype=np.bool_)
    intersection_points = np.zeros((1, len(tris), 3), dtype=np.float32)

    for j in range(tris.shape[0]):
        hit_boolean, hit_x, hit_y, hit_z = ray_intersects_tri(ray[0], tris[j])
        output[0, j] = hit_boolean
        if hit_boolean:
            intersection_points[0, j] = hit_x, hit_y, hit_z

    assert np.array_equal(output[0], expected_output), "Intersection output mismatch"

    if expected_intersection is not None:
        assert np.allclose(intersection_points[0, np.where(output[0])], expected_intersection, atol=1e-5), "Intersection point mismatch"

