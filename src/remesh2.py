import bpy
import sys

stl_no = sys.argv[5]

# clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# load an STL file
file_path = f'/projects/cnncae/didymus/src/projects/drivAer_r1025_n46/stls/{stl_no}.stl'
bpy.ops.import_mesh.stl(filepath=file_path)

# ensure the object is selected and active
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj


# add a Voxel Remesh modifier
remesh_modifier = obj.modifiers.new(name='Remesh', type='REMESH')
remesh_modifier.mode = 'VOXEL'
remesh_modifier.voxel_size = 0.2  # Decrease voxel size for higher resolution
remesh_modifier.use_remove_disconnected = True  # Remove disconnected parts

# apply the modifier
bpy.ops.object.modifier_apply(modifier='Remesh')

bpy.ops.export_mesh.stl(filepath=f'/projects/cnncae/didymus/src/projects/drivAer_r1025_n46/watertight-stls2/{stl_no}.stl')
