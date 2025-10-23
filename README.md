# ComfyUI-MeshCraft

**Complete 3D generation and manipulation package for ComfyUI**

Craft and transform 3D meshes in ComfyUI. Full Hunyuan 3D 2.1 integration with caching optimizations, advanced mesh processing, and experimental tools for prompt-based editing.

## Features

### 3D Generation (Powered by Hunyuan 3D 2.1)
- **Image-to-3D**: Generate high-quality meshes from single images
- **Multi-view Texturing**: AI-powered PBR texture generation
- **Model Caching**: 5-10x faster workflow execution with intelligent model caching
- **Batch Processing**: Process multiple images/meshes in one go

### Mesh Processing
- **Mesh Cleaning**: Remove disconnected floater geometry and degenerate faces
- **Face Reduction**: Intelligent remeshing using Instant Meshes algorithm
- **Normal Smoothing**: Smooth vertex normals for better shading
- **UV Unwrapping**: Prepare meshes for texturing
- **Format Export**: GLB, OBJ, STL support

## Installation

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "ComfyUI-MeshCraft"
3. Click Install

### Manual Installation

**Step 1: Install Hunyuan 3D (required for generation nodes)**
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1
cd ComfyUI-Hunyuan3d-2-1
# Follow their installation instructions
```

**Step 2: Install MeshCraft**
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft
cd ComfyUI-MeshCraft
pip install -r requirements.txt
```

**Step 3: Compile Extensions (for Hunyuan 3D nodes)**

The Hunyuan 3D nodes require two compiled extensions: `custom_rasterizer` (CUDA) and `DifferentiableRenderer` (C++).

**Option A: Use Precompiled Wheels (Recommended)**

If precompiled wheels are available for your Python version:

```bash
# Find your Python version
python --version

# Install custom_rasterizer (example for Python 3.12)
pip install hy3dpaint/custom_rasterizer/dist/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl

# Install DifferentiableRenderer (example for Python 3.12)
pip install hy3dpaint/DifferentiableRenderer/dist/mesh_inpaint_processor-0.0.0-cp312-cp312-linux_x86_64.whl
```

**Option B: Automated Compilation (Advanced Users)**

Use the automated compilation script that checks prerequisites and compiles both modules:

```bash
cd ComfyUI/custom_nodes/ComfyUI-MeshCraft
./compile_extensions.sh
```

**Prerequisites for compilation:**
- NVIDIA CUDA Toolkit (with `nvcc`)
- G++ compiler version 12+ (recommended)
- Python 3.10+
- PyTorch with CUDA support
- pybind11 (`pip install pybind11`)

**Script options:**
```bash
# Show help
./compile_extensions.sh --help

# Verbose mode (see detailed compilation output)
./compile_extensions.sh --verbose

# Force compilation even if some prerequisites are missing
./compile_extensions.sh --force

# Skip import verification after compilation
./compile_extensions.sh --skip-verify
```

The script will:
1. Check all prerequisites (nvcc, g++, Python, PyTorch CUDA, pybind11)
2. Check optional dependencies (Blender for UV unwrapping)
3. Compile custom_rasterizer (CUDA extension)
4. Compile DifferentiableRenderer (C++ extension)
5. Verify both modules can be imported
6. Report success or provide detailed error messages

**Option C: Manual Compilation (Expert Users)**

If you prefer manual control:

```bash
# Compile custom_rasterizer
cd hy3dpaint/custom_rasterizer
python setup.py install

# Compile DifferentiableRenderer
cd ../DifferentiableRenderer
python setup.py install
```

Restart ComfyUI after installation.

## Nodes

### MeshCraft Post-Process

Advanced mesh post-processing with multiple optimization options.

**Inputs:**
- `trimesh` (TRIMESH): Input mesh to process
- `remove_floaters` (BOOLEAN): Remove disconnected geometry components (default: True)
- `remove_degenerate_faces` (BOOLEAN): Remove invalid faces (default: True)
- `reduce_faces` (BOOLEAN): Apply face reduction (default: True)
- `max_facenum` (INT): Target maximum face count (default: 40000)
- `smooth_normals` (BOOLEAN): Smooth vertex normals (default: False)

**Outputs:**
- `trimesh` (TRIMESH): Processed mesh

**Example Use Cases:**
- Clean up generated 3D meshes before texturing
- Reduce polygon count for real-time applications
- Remove artifacts from 3D reconstruction
- Prepare meshes for physics simulations

## Requirements

### Core Requirements
- Python 3.10+
- ComfyUI
- See `requirements.txt` for Python package dependencies

### For Hunyuan 3D Nodes (Compilation)
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit (nvcc compiler)
- G++ compiler version 12+ (recommended)
- PyTorch with CUDA support
- pybind11

**Note**: Precompiled wheels are available in the `dist/` folders for common Python versions, which eliminates the need for compilation tools.

### Optional Requirements
- **Blender** (for UV unwrapping): Required for `Hy3DUVUnwrapper` node and texture generation with UV unwrapping
  - Install from: https://www.blender.org/download/
  - Must be available in system PATH as `blender` command
  - Used for Smart UV Project unwrapping via background subprocess

## Roadmap

- [ ] PyVista integration for parametric editing
- [ ] Multi-view editing with Flux
- [ ] Prompt-driven mesh deformation
- [ ] UV unwrapping utilities
- [ ] Mesh subdivision and smoothing

## Contributing

Contributions welcome! This is an experimental workshop for 3D mesh editing.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

**Mesh Processing Nodes**: MIT License (see LICENSE file)

**Hunyuan 3D Nodes**: Tencent Hunyuan 3D 2.1 Community License (see LICENSE_TENCENT_HUNYUAN)
- Free for research, education, and commercial use (<1M monthly active users)
- Geographic restrictions apply (excludes EU, UK, South Korea)
- Cannot be used to train competing AI models

## Credits

**MeshCraft**: Built for the ComfyUI community. Part of the scan2wall project.

**Hunyuan 3D Integration**: Based on [ComfyUI-Hunyuan3d-2-1](https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1) by visualbruno

**Modifications**:
- Added model caching for 5-10x faster reloads
- Memory management optimizations
- Package integration for full 3D pipeline

**Powered by**:
- Tencent Hunyuan 3D 2.1 (image-to-3D generation)
- Instant Meshes (remeshing algorithm)

## Support

- Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/issues)
- Discussions: [GitHub Discussions](https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/discussions)

## Acknowledgments

- Instant Meshes algorithm by Jakob et al.
- ComfyUI by comfyanonymous
- trimesh library