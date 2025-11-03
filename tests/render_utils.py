"""
Utility functions for rendering 3D meshes to images for visual verification.

This module provides tools to generate screenshots of GLB/STL/OBJ files
using Blender for high-quality rendering with material and texture preservation.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List


def _find_blender() -> Optional[str]:
    """
    Find Blender executable on the system.

    Returns:
        Path to Blender executable, or None if not found
    """
    blender_candidates = [
        "/usr/bin/blender",
        "/usr/local/blender/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        shutil.which("blender")
    ]

    for candidate in blender_candidates:
        if candidate and Path(candidate).exists():
            return candidate

    return None


def render_glb_to_image(
    mesh_path: str | Path,
    output_path: str | Path,
    image_size: Tuple[int, int] = (1024, 1024),
    camera_position: str = "isometric",
    background_color: str = "white",
    show_edges: bool = False,
) -> bool:
    """
    Render a 3D mesh file to a PNG image using Blender with material preservation.

    This function uses Blender's Cycles renderer to create high-quality renders
    that preserve materials and textures from GLB files. Works in headless
    environments.

    Args:
        mesh_path: Path to the mesh file (.glb, .stl, .obj)
        output_path: Path where the PNG screenshot will be saved
        image_size: Output image dimensions (width, height), default (1024, 1024)
        camera_position: Camera angle - "isometric", "front", "side", "top"
        background_color: Background color - "white", "black", or "transparent"
        show_edges: Whether to show mesh edges (not yet implemented for Blender)

    Returns:
        True if rendering succeeded, False otherwise

    Raises:
        RuntimeError: If Blender is not found on the system
    """
    mesh_path = Path(mesh_path)
    output_path = Path(output_path)

    if not mesh_path.exists():
        print(f"⚠️  Mesh file not found: {mesh_path}")
        return False

    # Find Blender
    blender_path = _find_blender()
    if not blender_path:
        raise RuntimeError(
            "Blender not found! Please install Blender 3.0+ to use GLB rendering.\n"
            "Download from: https://www.blender.org/download/"
        )

    # Map camera positions to Blender coordinates
    # Blender uses Z-up coordinate system
    camera_configs = {
        "isometric": {
            "location": (2.828, -2.828, 2.0),  # Diagonal angle showing 3 faces
            "rotation": (1.047, 0, 0.785),  # ~60° pitch, 45° yaw in radians
        },
        "front": {
            "location": (0, -4, 0),
            "rotation": (1.571, 0, 0),  # 90° pitch (looking along Y axis)
        },
        "side": {
            "location": (4, 0, 0),
            "rotation": (1.571, 0, 1.571),  # 90° pitch, 90° yaw
        },
        "top": {
            "location": (0, 0, 4),
            "rotation": (0, 0, 0),  # Looking down Z axis
        },
    }

    # Get camera configuration
    if camera_position not in camera_configs:
        print(f"⚠️  Unknown camera position '{camera_position}', using 'isometric'")
        camera_position = "isometric"

    cam_config = camera_configs[camera_position]
    cam_location = cam_config["location"]
    cam_rotation = cam_config["rotation"]

    # Map background color to Blender
    if background_color == "white":
        bg_rgb = (1.0, 1.0, 1.0, 1.0)
        transparent = False
    elif background_color == "black":
        bg_rgb = (0.0, 0.0, 0.0, 1.0)
        transparent = False
    elif background_color == "transparent":
        bg_rgb = (1.0, 1.0, 1.0, 0.0)
        transparent = True
    else:
        bg_rgb = (1.0, 1.0, 1.0, 1.0)
        transparent = False

    # Create Blender Python script
    blender_script = f'''
import bpy
import math

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import mesh (GLB with materials, or STL/OBJ)
mesh_path = "{mesh_path}"
if mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
    bpy.ops.import_scene.gltf(filepath=mesh_path)
elif mesh_path.endswith('.stl'):
    bpy.ops.wm.stl_import(filepath=mesh_path)
elif mesh_path.endswith('.obj'):
    bpy.ops.wm.obj_import(filepath=mesh_path)
else:
    print(f"Unsupported file format: {{mesh_path}}")
    exit(1)

# Find mesh objects in the scene
mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objects:
    print("No mesh objects found in file!")
    exit(1)

# Use the first mesh object
obj = mesh_objects[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Center and scale object
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)
max_dim = max(obj.dimensions) if max(obj.dimensions) > 0 else 1.0
obj.scale = (2.0 / max_dim, 2.0 / max_dim, 2.0 / max_dim)

# Ensure object has materials
if obj.data and hasattr(obj.data, 'materials'):
    if len(obj.data.materials) == 0:
        # Add default material if none exist
        mat = bpy.data.materials.new(name="Default")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        obj.data.materials.append(mat)

# Set up rendering
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = {image_size[0]}
scene.render.resolution_y = {image_size[1]}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = {str(transparent)}

# Cycles settings
scene.cycles.device = 'GPU'
scene.cycles.samples = 128
scene.cycles.use_denoising = False

# Set background
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg_node = scene.world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = {bg_rgb}
    bg_node.inputs[1].default_value = 1.0

# Create camera
camera_data = bpy.data.cameras.new(name="Camera")
camera_data.type = 'PERSP'
camera_data.lens = 50
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

# Position camera
camera.location = {cam_location}
camera.rotation_euler = {cam_rotation}

# Add 3-point lighting
# Key light
key_light_data = bpy.data.lights.new(name="KeyLight", type='SUN')
key_light_data.energy = 3.0
key_light = bpy.data.objects.new(name="KeyLight", object_data=key_light_data)
scene.collection.objects.link(key_light)
key_light.location = (5, -5, 10)

# Fill light
fill_light_data = bpy.data.lights.new(name="FillLight", type='SUN')
fill_light_data.energy = 1.5
fill_light = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
scene.collection.objects.link(fill_light)
fill_light.location = (-5, 5, 5)

# Back light
back_light_data = bpy.data.lights.new(name="BackLight", type='SUN')
back_light_data.energy = 1.0
back_light = bpy.data.objects.new(name="BackLight", object_data=back_light_data)
scene.collection.objects.link(back_light)
back_light.location = (0, 5, -5)

# Render
scene.render.filepath = "{output_path}"
bpy.ops.render.render(write_still=True)
'''

    # Write Blender script to temp file
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(blender_script)
            script_path = f.name

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run Blender
        result = subprocess.run(
            [blender_path, "--background", "--python", script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Clean up temp script
        Path(script_path).unlink(missing_ok=True)

        # Check if render succeeded
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            print(f"⚠️  Blender rendering failed for {mesh_path.name}")
            if result.stdout:
                print("stdout:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("stderr:", result.stderr[-500:])
            return False

    except subprocess.TimeoutExpired:
        print(f"⚠️  Blender rendering timed out for {mesh_path.name}")
        Path(script_path).unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"⚠️  Failed to render {mesh_path.name}: {e}")
        return False


def render_multiple_views(
    mesh_path: str | Path,
    output_dir: str | Path,
    views: Optional[List[str]] = None,
    **kwargs
) -> List[Path]:
    """
    Render a mesh from multiple camera angles.

    Args:
        mesh_path: Path to the mesh file
        output_dir: Directory where screenshots will be saved
        views: List of camera positions (default: ["isometric", "front", "side", "top"])
        **kwargs: Additional arguments passed to render_glb_to_image

    Returns:
        List of paths to generated screenshots
    """
    if views is None:
        views = ["isometric", "front", "side", "top"]

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_files = []

    for view in views:
        output_name = f"{mesh_path.stem}_{view}.png"
        output_path = output_dir / output_name

        success = render_glb_to_image(
            mesh_path,
            output_path,
            camera_position=view,
            **kwargs
        )

        if success:
            rendered_files.append(output_path)
            print(f"📸 Rendered {view} view: {output_path.name}")
        else:
            print(f"❌ Failed to render {view} view")

    return rendered_files


def create_contact_sheet(
    image_paths: List[Path],
    output_path: str | Path,
    grid_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Create a contact sheet (grid of images) from multiple screenshots.

    Args:
        image_paths: List of image file paths
        output_path: Path where the contact sheet will be saved
        grid_size: (rows, cols) for the grid. If None, automatically calculated.

    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        import math
    except ImportError:
        print("⚠️  PIL not installed. Install with: pip install Pillow")
        return False

    if not image_paths:
        print("⚠️  No images to combine")
        return False

    # Load images
    images = []
    for img_path in image_paths:
        if Path(img_path).exists():
            images.append(Image.open(img_path))
        else:
            print(f"⚠️  Image not found: {img_path}")

    if not images:
        print("⚠️  No valid images loaded")
        return False

    # Determine grid size
    if grid_size is None:
        num_images = len(images)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        grid_size = (rows, cols)

    rows, cols = grid_size

    # Get image size (assuming all images are the same size)
    img_width, img_height = images[0].size

    # Create contact sheet
    contact_width = img_width * cols
    contact_height = img_height * rows
    contact_sheet = Image.new('RGB', (contact_width, contact_height), 'white')

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        contact_sheet.paste(img, (x, y))

    # Save contact sheet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.save(output_path)
    print(f"📸 Contact sheet saved: {output_path}")

    return True
