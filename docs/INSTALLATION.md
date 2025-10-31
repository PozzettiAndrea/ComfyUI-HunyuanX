# ComfyUI-MeshCraft Installation Guide

## Quick Install (via ComfyUI-Manager) ⭐ Recommended

1. **Open ComfyUI Manager** in ComfyUI
2. **Search for "MeshCraft"**
3. **Click Install**
4. **Wait for CUDA compilation** (1-2 minutes, happens automatically)
5. **Restart ComfyUI**

✅ The `install.py` script automatically compiles the CUDA extension!

## Manual Installation

### 1. Clone the Repository

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft.git
cd ComfyUI-MeshCraft
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Compile CUDA Extension (Required for Texture Generation)

The CUDA rasterizer extension is needed for Hunyuan3D texture generation nodes.

**Option A: Automatic (via prestartup_script.py)**
- Just restart ComfyUI - the extension compiles automatically on first run!

**Option B: Manual Compilation**
```bash
cd hy3dpaint/custom_rasterizer
python setup.py install
cd ../..
```

**Verification:**
```python
python -c "import custom_rasterizer; print('✅ CUDA extension works!')"
```

### 4. Install System Dependencies (Linux Only)

For basic mesh preview nodes (uses pyglet/OpenGL):

```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa libosmesa6 xvfb
```

**Note:** Hunyuan rendering nodes use CUDA and don't need these!

## Troubleshooting

### CUDA Extension Won't Compile

**Error:** `ModuleNotFoundError: No module named 'custom_rasterizer'`

**Solutions:**

1. **Check CUDA is installed:**
   ```bash
   nvcc --version
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Install build tools:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential

   # Windows
   # Install Visual Studio Build Tools
   ```

3. **Manual compilation with verbose output:**
   ```bash
   cd hy3dpaint/custom_rasterizer
   python setup.py install --verbose
   ```

### OpenGL Library Not Found (Linux)

**Error:** `ImportError: Library "GLU" not found`

**Solution:**
```bash
sudo apt-get install -y libglu1-mesa libosmesa6
```

Then set environment variable:
```python
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
```

### GPU Out of Memory

**Error:** CUDA out of memory during rendering

**Solutions:**
- Reduce `resolution` parameter (try 256 or 512 instead of 1024)
- Close other GPU applications
- Use `--lowvram` flag when starting ComfyUI

## Dependencies

### Python Packages (installed via requirements.txt)
- `trimesh` - Mesh processing
- `pyglet` - OpenGL rendering (for preview nodes)
- `pymeshlab` - Advanced mesh operations
- `xatlas` - UV unwrapping
- `torch` - PyTorch (CUDA)
- Plus: diffusers, transformers, opencv-python, etc.

### System Libraries (Linux)
- `libglu1-mesa` - OpenGL utility library
- `libosmesa6` - Off-screen Mesa rendering
- `xvfb` - Virtual framebuffer (for headless servers)

### CUDA Extension (compiled from source)
- `custom_rasterizer` - Fast differentiable rasterizer for Hunyuan3D

## Verify Installation

```python
# Test basic imports
import trimesh
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test CUDA extension
import custom_rasterizer
print("✅ CUDA rasterizer loaded!")

# Test nodes can be imported
from custom_nodes.ComfyUI_MeshCraft import NODE_CLASS_MAPPINGS
print(f"✅ Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
```

## What Gets Installed

After installation, you'll have:

### Mesh Processing Nodes
- Load/Export mesh files (OBJ, GLB, PLY)
- Decimate/Optimize meshes
- UV unwrap
- Remesh operations

### Texture Generation Nodes
- Load Hunyuan multiview model
- Render conditioning maps (normals + positions)
- Generate PBR textures (albedo + metallic-roughness)

### Preview/Rendering Nodes
- TrimeshRender - Basic mesh preview
- Camera configuration
- Multi-view rendering

## Automatic Compilation Details

### prestartup_script.py
- Runs **before** ComfyUI starts
- Checks if `custom_rasterizer` is already compiled
- If not, compiles it automatically
- Only runs once (or when extension is missing)

### install.py
- Runs when installed **via ComfyUI-Manager**
- Installs requirements.txt dependencies
- Compiles CUDA extension
- Shows progress messages

Both scripts ensure the CUDA extension is ready without manual intervention!

## Uninstallation

### Via ComfyUI-Manager
1. Open Manager
2. Find "MeshCraft" in installed nodes
3. Click Uninstall

### Manual
```bash
cd /path/to/ComfyUI/custom_nodes
rm -rf ComfyUI-MeshCraft

# Also uninstall CUDA extension
pip uninstall custom-rasterizer -y
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux + NVIDIA GPU | ✅ Fully supported | Recommended |
| Windows + NVIDIA GPU | ✅ Fully supported | Requires VS Build Tools |
| macOS + Apple Silicon | ⚠️ Partial | CPU only, no CUDA extension |
| Linux + AMD GPU | ⚠️ Partial | ROCm may work, untested |

## Getting Help

- **GitHub Issues**: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/issues
- **ComfyUI Discord**: Ask in #custom-nodes channel
- **Check logs**: ComfyUI console shows compilation progress

## License

TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT (for Hunyuan3D components)

See LICENSE file for full details.
