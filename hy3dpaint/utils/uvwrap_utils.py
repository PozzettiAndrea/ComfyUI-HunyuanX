import trimesh
import time
import numpy as np
import subprocess
import tempfile
import os

def mesh_uv_wrap(mesh, angle_limit=66.0, island_margin=0.02):
    """
    Unwrap mesh using Blender's Smart UV Project.
    
    Args:
        mesh: trimesh mesh or scene
        angle_limit: angle threshold for creating seams (degrees)
        island_margin: spacing between UV islands (0-1)
    
    Returns:
        trimesh mesh with UV coordinates
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    a = time.time()
    
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
        input_path = f_in.name
        mesh.export(input_path)
    
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
        output_path = f_out.name
    
    try:
        # Blender script for UV unwrapping
        script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh (OBJ preserves geometry)
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Switch to edit mode and unwrap
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(
    angle_limit={np.radians(angle_limit)},
    island_margin={island_margin},
    area_weight=0.0,
    correct_aspect=True,
    scale_to_bounds=False
)
bpy.ops.object.mode_set(mode='OBJECT')

# Export with UVs
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=True,
    export_materials=False
)
"""
        
        result = subprocess.run([
            'blender', '--background', '--python-expr', script
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"Blender failed: {result.stderr}")
        
        # Load the unwrapped mesh
        unwrapped = trimesh.load(output_path, process=False)
        
        print(f"UV unwrapping took {time.time()-a} s!")
        
        return unwrapped
        
    finally:
        # Cleanup temp files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)