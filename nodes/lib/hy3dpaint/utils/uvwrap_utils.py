import trimesh
import time
import numpy as np
import subprocess
import tempfile
import os
import shutil
from pathlib import Path


def _find_blender():
    """
    Find Blender executable on the system.

    Checks in order:
    1. Local installation in _blender/ (downloaded by install.py)
    2. Environment variable BLENDER_PATH
    3. System installation (PATH or common locations)

    Returns:
        str: Path to Blender executable

    Raises:
        RuntimeError: If Blender not found
    """
    # Get the directory containing this file
    # Navigate from uvwrap_utils.py -> utils/ -> hy3dpaint/ -> lib/ -> nodes/ -> ComfyUI-HunyuanX/
    current_dir = Path(__file__).parent.parent.parent.parent.parent.parent
    local_blender_dir = current_dir / "_blender"

    # First, check for local Blender installation
    if local_blender_dir.exists():
        # Search for blender executable in _blender/
        blender_executables = []

        # Windows
        blender_executables.extend(list(local_blender_dir.rglob("blender.exe")))

        # Linux/macOS
        blender_executables.extend([
            p for p in local_blender_dir.rglob("blender")
            if p.is_file() and os.access(p, os.X_OK)
        ])

        if blender_executables:
            blender_path = str(blender_executables[0])
            print(f"[HunyuanX] Using local Blender: {blender_path}")
            return blender_path

    # Check environment variable
    env_blender = os.environ.get('BLENDER_PATH')
    if env_blender and os.path.exists(env_blender):
        print(f"[HunyuanX] Using Blender from BLENDER_PATH: {env_blender}")
        return env_blender

    # Fall back to system installation
    common_paths = [
        'blender',  # In PATH
        '/Applications/Blender.app/Contents/MacOS/Blender',  # macOS
        'C:\\Program Files\\Blender Foundation\\Blender\\blender.exe',  # Windows
        '/usr/bin/blender',  # Linux
        '/usr/local/bin/blender',  # Linux
    ]

    for path in common_paths:
        if shutil.which(path) or os.path.exists(path):
            print(f"[HunyuanX] Found system Blender: {path}")
            return path

    raise RuntimeError(
        "Blender not found. Please run 'python install.py' in the ComfyUI-HunyuanX directory to download Blender automatically,\n"
        "or install it manually from: https://www.blender.org/download/\n"
        "Texture generation requires Blender for UV unwrapping."
    )


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

        # Find Blender executable
        blender_path = _find_blender()

        result = subprocess.run([
            blender_path, '--background', '--python-expr', script
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