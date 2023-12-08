import trimesh

radius = 1.0
subdivisions = 5  # control the no. of elements

sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
sphere.export('sphere.stl')

print(f"No. of faces: {len(sphere.faces)}")
