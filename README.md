# ComfyUI-HunyuanX

**Hunyuan 3D 2.1 generation package for ComfyUI**

Integrates Tencent's Hunyuan 3D 2.1 for image-to-3D generation with PBR texturing, featuring both standard and modular pipelines.

## Features

### 3D Generation
- **Hunyuan 3D 2.1**: Image-to-3D with PBR texturing, model caching (5-10x speedup)
- **Standard Pipeline**: Quick generation with all-in-one nodes
- **Modular Pipeline**: Memory-efficient split DiT/VAE loading (andrea_nodes)

### Texture Generation
- **PBR Multiview**: Albedo + metallic-roughness textures
- **Conditioning Maps**: Normal/position map rendering
- **Inpainting**: Fill UV texture holes
- **Baking**: Multi-view texture baking

### Mesh Operations
- **Post-Processing**: Built-in mesh cleaning and optimization
- **UV Unwrapping**: Custom Hunyuan UV unwrapper
- **Export**: GLB, OBJ, STL, PLY, 3MF, DAE, FBX

### Interpolation
- **Latent Blending**: Hunyuan3D latents, DINO embeddings
- **SLERP/LERP**: Smooth transitions between meshes/images

## Installation

### Quick Install
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_USERNAME/ComfyUI-HunyuanX
cd ComfyUI-HunyuanX
pip install -r requirements.txt
```

### Via ComfyUI Manager
1. Open ComfyUI Manager
2. Search for "ComfyUI-HunyuanX"
3. Click Install

**Requirements**: Python 3.10+, PyTorch 2.5-2.8, NVIDIA GPU with CUDA (recommended 24GB+ VRAM)

---

## All Nodes (64 total)

### Mesh Processing (3 nodes)
| Node | Purpose |
|------|---------|
| `MeshCraftPostProcess` | Floater removal, face reduction, normal smoothing (fixed order) |
| `MeshCraft_GeomOperations` | **NEW**: Flexible 1-10 operation slots with intermediate outputs |
| `MeshCraftUVUnwrap` | UV unwrapping (xatlas or Blender) |

### Hunyuan 3D - Monolithic (13 nodes)
| Node | Purpose |
|------|---------|
| `Hy3DMeshGenerator` | Generate 3D mesh from image (with caching) |
| `Hy3DMultiViewsGenerator` | Generate PBR multiview renders (albedo, MR, normals, positions) |
| `Hy3DBakeMultiViews` | Bake multiview images to UV texture |
| `Hy3DInPaint` | Fill texture holes |
| `Hy3D21CameraConfig` | Configure camera azimuths, elevations, weights |
| `Hy3D21VAELoader` | Load VAE with caching |
| `Hy3D21VAEDecode` | Decode latents to mesh |
| `Hy3D21LoadImageWithTransparency` | Load RGBA image |
| `Hy3D21PostprocessMesh` | Hunyuan post-processor (pymeshlab-based) |
| `Hy3D21ExportMesh` | Export mesh to GLB/OBJ/PLY/STL/3MF/DAE |
| `Hy3D21MeshUVWrap` | UV unwrap mesh |
| `Hy3D21LoadMesh` | Load mesh from GLB |
| `Hy3D21ImageWithAlphaInput` | Combine image + mask → RGBA |

### Hunyuan 3D - Modular (9 nodes)
*Memory-efficient split pipeline (load DiT/VAE separately)*
| Node | Purpose |
|------|---------|
| `PrepareImageForDINO` | Recenter, resize, normalize for DINO |
| `LoadDinoModel` | Load DINO v2 (small/base/large/giant) |
| `EncodeDINO` | Encode image to DINO embeddings |
| `LoadHunyuanDiT` | Load DiT model only (saves ~2GB VRAM) |
| `LoadHunyuanVAE` | Load VAE decoder separately |
| `Hy3DGenerateLatents` | Generate 3D latents from DINO embeddings |
| `Hy3DDecodeLatents` | Decode latents to mesh |
| `Hy3DImageToLatents` | All-in-one convenience node |
| `PreviewTrimesh` | Interactive 3D preview (PyVista) |

### Texture Generation / Rendering (7 nodes)
| Node | Purpose |
|------|---------|
| `LoadHunyuanMultiViewModel` | Load multiview diffusion model for PBR |
| `RenderConditioningMaps` | Render normal + position maps |
| `RenderRGBMultiview` | Blender rendering (Cycles/Eevee) |
| `GenerateMultiviewPBR` | Generate albedo + MR textures |
| `BakeMultiviewTextures` | Geometric projection to UV space |
| `InpaintTextures` | OpenCV inpainting (NS/Telea) |
| `ApplyAndSaveTexturedMesh` | Apply textures and export OBJ + GLB |

### TRELLIS (17 nodes)
**Loaders (3)**
| Node | Purpose |
|------|---------|
| `Load_DinoV2_Model` | Load DINO v2 for TRELLIS |
| `Load_CLIP_Trellis` | Load CLIP for text conditioning |
| `Load_Trellis_Model` | Load TRELLIS pipeline (image/text) |

**Granular Pipeline (9)**
| Node | Purpose |
|------|---------|
| `Trellis_Image_Preprocessor` | Preprocess image (remove background, recenter) |
| `Trellis_Image_Conditioning` | Encode image to conditioning |
| `Trellis_Text_Conditioning` | Encode text to conditioning |
| `Trellis_SparseStructure_Sampler` | Sample sparse voxel structure |
| `Trellis_SLAT_Sampler` | Sample SLAT embeddings |
| `Trellis_SLAT_Decoder` | Decode to Gaussian/mesh/radiance field |
| `Trellis_SLAT_Visualizer` | Visualize SLAT as 3D point cloud |
| `Trellis_Export_GLB` | Export mesh to GLB |
| `Trellis_Export_PLY` | Export Gaussian splats to PLY |
| `Trellis_Render_Video` | Render 360° turntable video |

**Utilities (1)**
| Node | Purpose |
|------|---------|
| `Trellis_multiimage_loader` | Load 2-3 images for multi-view fusion |

### Interpolation (3 nodes)
| Node | Purpose |
|------|---------|
| `VAELatentInterpolation` | Interpolate between images in VAE latent space (SLERP/LERP) |
| `Hunyuan3DLatentInterpolation` | Morph between 3D meshes |
| `DINOEmbeddingInterpolation` | Semantic image transitions |

---

## Key Improvements

### ✅ Model Caching
Hunyuan nodes cache models between runs for 5-10x speedup

### ✅ Lazy Loading
Startup time reduced from ~26s to <2s via deferred imports

### ✅ Granular Pipelines
- **TRELLIS**: Split into 9 nodes for fine control
- **Hunyuan**: Modular pipeline with separate DiT/VAE loading

### ✅ Flexible Geometric Operations (NEW)
`MeshCraft_GeomOperations` allows:
- 1-10 operation slots (configurable via slider)
- Arbitrary operation order
- Intermediate outputs for each operation
- Operations: remove_floaters, remove_degenerate, reduce_faces, smooth_normals, laplacian_smooth, ensure_manifold

### ✅ Multi-View Fusion
TRELLIS supports combining 2-3 views (front/back/side) for better reconstruction (ideal for pottery fragments)

---

## Example Workflows

Check the `workflows/` directory for:
- `hunyuan-i2m.json` - Hunyuan image-to-3D
- `trellis-i2m.json` - TRELLIS image-to-3D
- `trellis-t2m.json` - TRELLIS text-to-3D

---

## Documentation

- **Installation**: See `docs/INSTALLATION.md` for detailed setup
- **Hunyuan Guide**: `docs/README_HUNYUAN.md`
- **TRELLIS Guide**: `docs/TRELLIS_QUICKSTART.md`
- **Interpolation**: `docs/INTERPOLATION_USAGE.md`

---

## Requirements

**Core:**
- Python 3.10+
- PyTorch 2.5-2.8 (for Kaolin compatibility)
- NVIDIA GPU with CUDA (for full features)
- See `requirements.txt` for full list

**Optional:**
- **Blender**: For Smart UV unwrapping and RGB multiview rendering
- **flash-attn**: 10-20% faster inference (requires compilation)
- **spconv**: For TRELLIS sparse convolution (auto-installs)

---

## Roadmap

- [x] PyVista integration (✅ done - `PreviewTrimesh`)
- [x] UV unwrapping utilities (✅ done - 2 nodes)
- [x] Flexible geometric operations (✅ done - `MeshCraft_GeomOperations`)
- [ ] Batch processing improvements
- [ ] Additional texture editing tools

---

## License

See individual model licenses:
- **Hunyuan 3D**: Tencent Hunyuan Community License (<1M MAU)
- **TRELLIS**: Microsoft Research License
- **MeshCraft code**: MIT License

---

## Contributing

Issues and PRs welcome at: https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft

---

**Version**: 0.4.0 | **Total Nodes**: 64
