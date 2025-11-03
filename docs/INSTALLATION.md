# ComfyUI-MeshCraft Installation Guide

## Quick Install (Recommended)

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft
cd ComfyUI-MeshCraft
python install.py
```

The `install.py` script automatically:
- Installs Python dependencies
- Compiles CUDA extensions (if CUDA available)
- Installs Kaolin from NVIDIA wheels
- Validates PyTorch version

Then restart ComfyUI.

## Via ComfyUI Manager

1. Open ComfyUI Manager
2. Search for "MeshCraft"
3. Click Install
4. Restart ComfyUI

## System Requirements

**Minimum:**
- Python 3.10+
- PyTorch 2.5-2.8 (for Kaolin compatibility)
- 8GB RAM
- 4GB VRAM (CPU fallback available for some nodes)

**Recommended:**
- NVIDIA GPU with 12GB+ VRAM
- CUDA Toolkit 11.8+ (for extension compilation)
- 16GB+ RAM

## Optional Dependencies

**Blender** (for UV unwrapping and RGB rendering):
```bash
# Install from https://www.blender.org/download/
# Make sure `blender` command is in PATH
```

**flash-attn** (10-20% faster, requires 5-10 min compilation):
```bash
pip install flash-attn --no-build-isolation
```

**spconv** (for TRELLIS, auto-installs when needed):
- Installed automatically by `install.py` if CUDA available

## Troubleshooting

### CUDA Extension Compilation Fails
- Verify CUDA Toolkit is installed: `nvcc --version`
- Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
- Try precompiled wheels (if available for your Python version)

### PyTorch Version Mismatch
- Kaolin requires PyTorch 2.5-2.8
- Reinstall: `pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu118`

### Import Errors
- Run `install.py` again: `python install.py`
- Check that all dependencies installed: `pip install -r requirements.txt`

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux + NVIDIA GPU** | ✅ Fully supported | Recommended platform |
| **Windows + NVIDIA GPU** | ✅ Supported | Requires VS Build Tools for compilation |
| **macOS (Apple Silicon)** | ⚠️ Partial | CPU-only, no CUDA extensions |
| **AMD GPU** | ❌ Untested | ROCm might work but not officially supported |

## What Gets Installed

**Python Packages** (59 total):
- Core: `trimesh`, `pymeshlab`, `open3d`, `xatlas`
- AI Models: `transformers`, `diffusers`, `timm`
- 3D Libraries: `kaolin`, `nvdiffrast` (optional), `pyvista`
- Rendering: `rembg`, `opencv-python`

**CUDA Extensions** (compiled):
- `custom_rasterizer` - Differentiable rendering for texturing
- `mesh_inpaint_processor` - Texture inpainting

**Models** (auto-downloaded on first use):
- Hunyuan 3D 2.1: ~4GB (from HuggingFace)
- TRELLIS: 2-3GB (from HuggingFace)
- DINO v2: ~300MB per model size

## Verification

Test that installation worked:
```python
# Test imports
python -c "from nodes.geom_operations_node import MeshCraft_GeomOperations; print('✅ Nodes load OK')"

# Test CUDA extension (if applicable)
python -c "import custom_rasterizer; print('✅ CUDA extension OK')"
```

## Need Help?

- Check the main README.md
- Review example workflows in `workflows/`
- Open an issue: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/issues
