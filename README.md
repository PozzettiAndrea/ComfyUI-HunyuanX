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

- Python 3.10+
- ComfyUI
- trimesh
- numpy
- pynanoinstantmeshes

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