"""
ComfyUI-MeshCraft

Complete 3D generation and manipulation package for ComfyUI.

Features:
- Full Hunyuan 3D 2.1 integration with model caching (~5-10x faster reloads)
- Microsoft TRELLIS 3D generation with granular pipeline control
- Advanced mesh post-processing (clean, optimize, reduce faces)
- UV unwrapping and texture baking
- Batch processing capabilities
- Hybrid workflows (TRELLIS geometry + Hunyuan textures)

Repository: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft
"""

import sys
import os

# Add lib/ to sys.path so hy3dshape and hy3dpaint are importable as top-level modules
# This allows YAML configs to reference them directly (e.g., hy3dshape.hy3dshape.models)
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, "lib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Import from organized nodes/ directory
from .nodes import (
    MESH_NODES,
    MESH_DISPLAY,
    INTERP_NODES,
    INTERP_DISPLAY,
    HUNYUAN_NODES,
    HUNYUAN_DISPLAY,
    ANDREA_NODES,
    ANDREA_DISPLAY,
    RENDERING_NODES,
    RENDERING_DISPLAY,
    TRELLIS_NODES,
    TRELLIS_DISPLAY,
)

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
num_nodes = len(NODE_CLASS_MAPPINGS)
print(f"✅ ComfyUI-MeshCraft: Loaded {num_nodes} nodes")
print(f"   • Mesh nodes: {len(MESH_NODES)}")
print(f"   • Interpolation nodes: {len(INTERP_NODES)}")
print(f"   • Hunyuan nodes: {len(HUNYUAN_NODES)}")
print(f"   • Andrea nodes: {len(ANDREA_NODES)}")
print(f"   • Rendering nodes: {len(RENDERING_NODES)}")
print(f"   • TRELLIS nodes: {len(TRELLIS_NODES)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.4.0"  # Major reorganization + TRELLIS granular pipeline
