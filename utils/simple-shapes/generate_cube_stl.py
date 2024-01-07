import trimesh

def upsample_cube(target_faces):
    # create a unit cube
    size = 1.0
    cube = trimesh.creation.box(extents=[size, size, size])

    # subdivide the cube until the number of faces reaches the target
    while len(cube.faces) < target_faces:
        cube = cube.subdivide()

    return cube

target_faces = 12

upsampled_cube = upsample_cube(target_faces)

# Export the upsampled cube to an STL file
upsampled_cube.export('cube.stl')
