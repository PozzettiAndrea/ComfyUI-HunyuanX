"""
ComfyUI-MeshCraft

Advanced mesh manipulation and optimization nodes for ComfyUI.

Features:
- Mesh cleaning (remove floaters, degenerate faces)
- Face reduction using Instant Meshes
- Normal smoothing
- Optimized for 3D-to-3D editing workflows

Repository: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "0.1.0"
