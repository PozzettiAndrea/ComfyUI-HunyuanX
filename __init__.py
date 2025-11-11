"""
ComfyUI-HunyuanX

Hunyuan 3D 2.1 generation package for ComfyUI.

Features:
- Full Hunyuan 3D 2.1 integration with model caching (~5-10x faster reloads)
- Standard Hunyuan pipeline for quick generation
- Modular/granular pipeline control (andrea_nodes)
- Multi-view texture generation and baking
- Texture inpainting and completion
- UV unwrapping and mesh post-processing
- Batch processing capabilities

Repository: https://github.com/YOUR_USERNAME/ComfyUI-HunyuanX
"""

import sys
import os

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# This prevents relative import errors when pytest collects test modules
if 'pytest' not in sys.modules:
    # Add nodes/lib/ to sys.path so hy3dshape and hy3dpaint are importable as top-level modules
    # This allows YAML configs to reference them directly (e.g., hy3dshape.hy3dshape.models)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(current_dir, "nodes", "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    # Import Hunyuan nodes
    from .nodes import (
        HUNYUAN_NODES,
        HUNYUAN_DISPLAY,
        ANDREA_NODES,
        ANDREA_DISPLAY,
        RENDERING_NODES,
        RENDERING_DISPLAY,
    )

    # Merge all Hunyuan nodes
    NODE_CLASS_MAPPINGS = {
        **HUNYUAN_NODES,
        **ANDREA_NODES,
        **RENDERING_NODES,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        **HUNYUAN_DISPLAY,
        **ANDREA_DISPLAY,
        **RENDERING_DISPLAY,
    }

    # Print status
    num_nodes = len(NODE_CLASS_MAPPINGS)
    print(f"✅ ComfyUI-HunyuanX: Loaded {num_nodes} nodes")
    print(f"   • Hunyuan nodes: {len(HUNYUAN_NODES)}")
    print(f"   • Andrea nodes: {len(ANDREA_NODES)}")
    print(f"   • Rendering nodes: {len(RENDERING_NODES)}")
else:
    # During testing, set empty mappings
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"  # Hunyuan 3D 2.1 nodes for ComfyUI
