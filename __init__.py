"""
ComfyUI-MeshCraft

Complete 3D generation and manipulation package for ComfyUI.

Features:
- Full Hunyuan 3D 2.1 integration with model caching (~5-10x faster reloads)
- Microsoft TRELLIS 3D generation with multi-view support
- Advanced mesh post-processing (clean, optimize, reduce faces)
- UV unwrapping and texture baking
- Batch processing capabilities
- Hybrid workflows (TRELLIS geometry + Hunyuan textures)

Repository: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft
"""

from .nodes import NODE_CLASS_MAPPINGS as MESH_NODES, NODE_DISPLAY_NAME_MAPPINGS as MESH_DISPLAY
from .interpolation_nodes import NODE_CLASS_MAPPINGS as INTERP_NODES, NODE_DISPLAY_NAME_MAPPINGS as INTERP_DISPLAY

# Try to load Hunyuan nodes
try:
    from .hunyuan_nodes import NODE_CLASS_MAPPINGS as HUNYUAN_NODES, NODE_DISPLAY_NAME_MAPPINGS as HUNYUAN_DISPLAY
    from .andrea_nodes import NODE_CLASS_MAPPINGS as ANDREA_NODES, NODE_DISPLAY_NAME_MAPPINGS as ANDREA_DISPLAY
    from .rendering_nodes import NODE_CLASS_MAPPINGS as RENDERING_NODES, NODE_DISPLAY_NAME_MAPPINGS as RENDERING_DISPLAY
    hunyuan_loaded = True
except ImportError as e:
    print(f"⚠️  ComfyUI-MeshCraft: Could not load Hunyuan nodes: {e}")
    HUNYUAN_NODES = {}
    HUNYUAN_DISPLAY = {}
    ANDREA_NODES = {}
    ANDREA_DISPLAY = {}
    RENDERING_NODES = {}
    RENDERING_DISPLAY = {}
    hunyuan_loaded = False

# Try to load TRELLIS nodes
try:
    from .trellis_nodes import NODE_CLASS_MAPPINGS as TRELLIS_NODES, NODE_DISPLAY_NAME_MAPPINGS as TRELLIS_DISPLAY
    trellis_loaded = True
except ImportError as e:
    print(f"⚠️  ComfyUI-MeshCraft: Could not load TRELLIS nodes: {e}")
    TRELLIS_NODES = {}
    TRELLIS_DISPLAY = {}
    trellis_loaded = False

# Merge all available nodes
NODE_CLASS_MAPPINGS = {
    **MESH_NODES,
    **INTERP_NODES,
    **HUNYUAN_NODES,
    **ANDREA_NODES,
    **RENDERING_NODES,
    **TRELLIS_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MESH_DISPLAY,
    **INTERP_DISPLAY,
    **HUNYUAN_DISPLAY,
    **ANDREA_DISPLAY,
    **RENDERING_DISPLAY,
    **TRELLIS_DISPLAY,
}

# Print status
loaded_modules = ["mesh", "interpolation"]
if hunyuan_loaded:
    loaded_modules.extend(["hunyuan", "andrea", "rendering"])
if trellis_loaded:
    loaded_modules.append("trellis")

print(f"✅ ComfyUI-MeshCraft: Loaded {' + '.join(loaded_modules)} nodes")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.3.0"  # Bump version for TRELLIS integration
