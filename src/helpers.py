import numpy as np

def compute_bbox(stl_mesh, buffer_percent=5):
    """
    Compute the bounding box of an STL file with a buffer on all sides.
    """
    # reshape the points array to a (n, 3, 3) array 
    reshaped_points = stl_mesh.points.reshape(-1, 3, 3)

    # find the min and max points
    min_point = np.min(reshaped_points, axis=(0, 1))
    max_point = np.max(reshaped_points, axis=(0, 1))

    # calculate the buffer size
    buffer_size = ((max_point - min_point) * buffer_percent / 100)

    # apply the buffer to the bounding box
    x_min, y_min, z_min = min_point - buffer_size
    x_max, y_max, z_max = max_point + buffer_size

    return x_min, x_max, y_min, y_max, z_min, z_max
