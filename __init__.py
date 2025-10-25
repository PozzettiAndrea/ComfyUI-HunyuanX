"""
ComfyUI-MeshCraft

Complete 3D generation and manipulation package for ComfyUI.

Features:
- Full Hunyuan 3D 2.1 integration with model caching (~5-10x faster reloads)
- Advanced mesh post-processing (clean, optimize, reduce faces)
- UV unwrapping and texture baking
- Batch processing capabilities
- Ready for PyVista integration and custom pipeline extensions

Repository: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft
"""

from .nodes import NODE_CLASS_MAPPINGS as MESH_NODES, NODE_DISPLAY_NAME_MAPPINGS as MESH_DISPLAY
from .interpolation_nodes import NODE_CLASS_MAPPINGS as INTERP_NODES, NODE_DISPLAY_NAME_MAPPINGS as INTERP_DISPLAY

try:
    from .hunyuan_nodes import NODE_CLASS_MAPPINGS as HUNYUAN_NODES, NODE_DISPLAY_NAME_MAPPINGS as HUNYUAN_DISPLAY
    from .andrea_nodes import NODE_CLASS_MAPPINGS as ANDREA_NODES, NODE_DISPLAY_NAME_MAPPINGS as ANDREA_DISPLAY
    # Merge all the mappings
    NODE_CLASS_MAPPINGS = {**MESH_NODES, **INTERP_NODES, **HUNYUAN_NODES, **ANDREA_NODES}
    NODE_DISPLAY_NAME_MAPPINGS = {**MESH_DISPLAY, **INTERP_DISPLAY, **HUNYUAN_DISPLAY, **ANDREA_DISPLAY}
    print("✅ ComfyUI-MeshCraft: Loaded all nodes (mesh + interpolation + hunyuan + andrea modular)")
except ImportError as e:
    print(f"⚠️  ComfyUI-MeshCraft: Could not load Hunyuan nodes. Install ComfyUI-Hunyuan3d-2-1 first.")
    print(f"   Error: {e}")
    # Fall back to mesh processing and interpolation nodes
    NODE_CLASS_MAPPINGS = {**MESH_NODES, **INTERP_NODES}
    NODE_DISPLAY_NAME_MAPPINGS = {**MESH_DISPLAY, **INTERP_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.2.0"  # Bump version for Hunyuan integration
