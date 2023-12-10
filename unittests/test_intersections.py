import numpy as np
from numba import cuda
import sys
sys.path.append('../src') 
from kernels import ray_intersects_tri


Ray = np.dtype([('origin', np.float32, (3,)), ('direction', np.float32, (3,))])

def test_ray_triangle_intersection():
    """Test ray tracing using two triangles: one that intersects and one
       that doesn't
    """
    ray = np.array([([0, 0, 0], [0, 1, 0])], dtype=Ray)  
    triangles = np.array([([1, 1, 0], [1, -1, 0], [1, 0, 1]),  # not intersecting
                          ([-1, 0, 0], [1, 0, 0], [0, 1, 1])], # intersecting
                         dtype=np.float32)

    ray_device = cuda.to_device(ray)
    triangles_device = cuda.to_device(triangles)
    output = np.zeros((1, 2), dtype=np.bool_)  # For storing results
    output_device = cuda.to_device(output)

    @cuda.jit
    def test_kernel(ray, triangles, output):
        for j in range(triangles.shape[0]):
            intersects, _ = ray_intersects_tri(ray[0], triangles[j])
            output[0, j] = intersects

    # run test kernel
    test_kernel[1, 1](ray_device, triangles_device, output_device)
    output_device.copy_to_host(output)

    if output[0, 0] == False and output[0, 1] == True:
        print("Test passed")
    else:
        print("Test failed")

test_ray_triangle_intersection()
