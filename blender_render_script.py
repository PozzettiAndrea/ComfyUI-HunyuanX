"""
Blender rendering script for ComfyUI-MeshCraft
Renders mesh from multiple orthographic viewpoints matching Hunyuan3D-2.1 training pipeline
"""

import bpy
import json
import sys
import math
from mathutils import Vector
from pathlib import Path

# Get arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:]
mesh_path = Path(argv[0])
config_path = Path(argv[1])

print(f"Loading config from: {config_path}")

# Load config
with open(config_path) as f:
    config = json.load(f)

print(f"Config loaded: {len(config['azimuths'])} views to render")

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import mesh
print(f"Importing mesh from: {mesh_path}")
bpy.ops.wm.obj_import(filepath=str(mesh_path))

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
print(f"Imported object: {obj.name}")

# Center and normalize object
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 0)

# Scale to unit cube
max_dim = max(obj.dimensions)
if max_dim > 0:
    obj.scale = (1.15 / max_dim, 1.15 / max_dim, 1.15 / max_dim)

print(f"Object scaled: {obj.scale}")

# Add material for better lighting
mat = bpy.data.materials.new(name="MeshMaterial")
mat.use_nodes = True
if len(obj.data.materials) == 0:
    obj.data.materials.append(mat)
else:
    obj.data.materials[0] = mat

# Set up rendering
scene = bpy.context.scene
scene.render.engine = config["engine"]
scene.render.resolution_x = config["resolution"]
scene.render.resolution_y = config["resolution"]
scene.render.film_transparent = (config["background"] == "transparent")

print(f"Render engine: {config['engine']}, resolution: {config['resolution']}")

if config["engine"] == "CYCLES":
    scene.cycles.samples = config["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.device = 'GPU'  # Use GPU if available

# Create camera
camera_data = bpy.data.cameras.new(name="Camera")
camera_data.type = 'ORTHO'
camera_data.ortho_scale = config["ortho_scale"] * 2.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

# Create lighting (3-point setup)
light_data = bpy.data.lights.new(name="KeyLight", type='SUN')
light_data.energy = 2.0
key_light = bpy.data.objects.new(name="KeyLight", object_data=light_data)
scene.collection.objects.link(key_light)
key_light.location = (5, 5, 10)

light_data2 = bpy.data.lights.new(name="FillLight", type='SUN')
light_data2.energy = 1.0
fill_light = bpy.data.objects.new(name="FillLight", object_data=light_data2)
scene.collection.objects.link(fill_light)
fill_light.location = (-5, -5, 5)

print("Lights and camera created")

# Render each view
output_dir = Path(config["output_dir"])
for i, (azim, elev) in enumerate(zip(config["azimuths"], config["elevations"])):
    print(f"Rendering view {i}: azim={azim}°, elev={elev}°")

    # Convert to radians
    azim_rad = math.radians(azim)
    elev_rad = math.radians(elev)

    # Calculate camera position (spherical coordinates)
    distance = 1.5
    x = distance * math.cos(elev_rad) * math.sin(azim_rad)
    y = distance * math.sin(elev_rad)
    z = distance * math.cos(elev_rad) * math.cos(azim_rad)

    # Set camera position and look at object
    camera.location = (x, y, z)
    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Render
    output_path = str(output_dir / f"render_{i:03d}.png")
    scene.render.filepath = output_path
    print(f"Rendering to: {output_path}")
    bpy.ops.render.render(write_still=True)
    print(f"Rendered view {i}")

print(f"Successfully rendered {len(config['azimuths'])} views")
