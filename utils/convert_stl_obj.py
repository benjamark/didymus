###############################################################################
# Utility to convert from STL to OBJ and vice-versa.                          #
###############################################################################

import trimesh

input_file = 'shapenet/1.stl'
output_file = '1.obj'

mesh = trimesh.load(input_file)
if mesh.is_watertight:
    print("The mesh is watertight. Proceeding with conversion.")
else:
    print("The mesh is not watertight. Converted file might not be watertight.")
mesh.export(output_file)
