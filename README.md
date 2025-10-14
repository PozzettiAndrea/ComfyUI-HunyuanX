# ComfyUI-MeshCraft

**Advanced mesh manipulation and optimization nodes for ComfyUI**

Craft and transform 3D meshes in ComfyUI. Experimental tools for prompt-based editing, mesh optimization, and parametric sculpting.

## Features

- **Mesh Cleaning**: Remove disconnected floater geometry and degenerate faces
- **Face Reduction**: Intelligent remeshing using Instant Meshes algorithm
- **Normal Smoothing**: Smooth vertex normals for better shading
- **Optimized Performance**: Fast processing for 3D-to-3D editing workflows

## Installation

### Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "ComfyUI-MeshCraft"
3. Click Install

### Manual Installation

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

MIT License - see LICENSE file for details

## Credits

Built for the ComfyUI community.

Part of the scan2wall project tooling.

## Support

- Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/issues)
- Discussions: [GitHub Discussions](https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/discussions)

## Acknowledgments

- Instant Meshes algorithm by Jakob et al.
- ComfyUI by comfyanonymous
- trimesh library