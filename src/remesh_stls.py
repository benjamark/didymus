import bpy
import sys
import os

stl_no = sys.argv[5]

# REMEMBER TO CHANGE PATHS; DOES NOT MIRROR CONFIG

# clear all existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

output_dir = '/projects/cnncae/didymus/src/projects/drivAer_r2049_n49/surfer-stls'
os.makedirs(output_dir, exist_ok=True)

file_path = f'/projects/cnncae/didymus/src/projects/drivAer_r2049_n49/stls/{stl_no}.stl'
bpy.ops.import_mesh.stl(filepath=file_path)

output_path = f'{output_dir}/{stl_no}.stl'
if os.path.exists(output_path):
    print(f"Output file {output_path} already exists. Skipping.")
else:
    # check if any objects were imported
    if not bpy.context.selected_objects:
        print("Error: No objects were imported.")
    else:
        obj = bpy.context.selected_objects[0]

        if obj.type != 'MESH':
            print("Error: Imported object is not a mesh.")
        else:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.object.mode_set(mode='OBJECT')

            modifier = obj.modifiers.new(name='Decimate', type='DECIMATE')

            modifier.ratio = 0.025
            modifier.use_collapse_triangulate = True

            bpy.ops.object.modifier_apply(modifier='Decimate')

            remesh_modifier = obj.modifiers.new(name='Remesh', type='REMESH')
            remesh_modifier.mode = 'VOXEL'
            remesh_modifier.voxel_size = 2.0  # decrease voxel size for higher resolution
            remesh_modifier.use_remove_disconnected = True  # Remove disconnected parts

            bpy.ops.object.modifier_apply(modifier='Remesh')

            output_path = f'{output_dir}/{stl_no}.stl'
            bpy.ops.export_mesh.stl(filepath=output_path)
