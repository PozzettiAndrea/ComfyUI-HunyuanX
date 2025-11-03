# TRELLIS Quick Start Guide

Microsoft TRELLIS - Structured 3D latents for multi-view 3D generation.

## Overview

TRELLIS excels at:
- **Multi-view fusion** - Combine 2-3 images (front/back/side)
- **Fragmented objects** - Better than dense voxels for incomplete geometry
- **Multiple outputs** - Mesh, Gaussian splats, radiance fields
- **Text-to-3D** - Generate from text descriptions

**Ideal for**: Pottery fragments, archaeological reconstruction, multi-view captures

**Paper**: [Structured 3D Latents](https://arxiv.org/abs/2412.01506) (CVPR 2025)

---

## Installation

TRELLIS dependencies auto-install when you run `python install.py`.

**Optional** (10-20% speedup):
```bash
pip install flash-attn --no-build-isolation
```

---

## Basic Workflow (Image-to-3D)

### 1. Load Models
```
Load_DinoV2_Model → dinov2_model
Load_Trellis_Model (model_name="microsoft/TRELLIS-image-large") → trellis_model
```

### 2. Granular Pipeline
```
LoadImage → image
  ↓
Trellis_Image_Preprocessor → preprocessed_image
├─ remove_background: true
└─ recenter: true
  ↓
Trellis_Image_Conditioning → conditioning
├─ dinov2_model
└─ preprocessed_image
  ↓
Trellis_SparseStructure_Sampler → sparse_latents
├─ trellis_model
├─ conditioning
├─ seed: 42
├─ cfg: 7.5
└─ steps: 12
  ↓
Trellis_SLAT_Sampler → slat_latents
├─ trellis_model
├─ sparse_latents
├─ cfg: 3.0
└─ steps: 12
  ↓
Trellis_SLAT_Decoder → outputs
├─ slat_latents
├─ formats: ["mesh", "gaussian"]
└─ texture_size: 1024
  ↓
Trellis_Export_GLB → glb_path
```

---

## Multi-View Fusion (Pottery Reconstruction)

Combine 2-3 views for better geometry:

```
LoadImage (front) ─┐
LoadImage (back)  ─┼→ Trellis_multiimage_loader → combined_image
LoadImage (side)  ─┘
  ↓
Trellis_Image_Preprocessor
  ↓
Trellis_Image_Conditioning
  ↓
Trellis_SparseStructure_Sampler
├─ multi_image: true
└─ multiimage_algo: "multidiffusion"  # or "stochastic"
  ↓
... (continue as single-image pipeline)
```

**Algorithms**:
- `multidiffusion`: Averages features (smoother, recommended)
- `stochastic`: Random view selection (faster, more varied)

---

## Text-to-3D

```
Load_CLIP_Trellis → clip_model
Load_Trellis_Model (model_name="microsoft/TRELLIS-text-large") → trellis_model
  ↓
Trellis_Text_Conditioning → conditioning
├─ clip_model
└─ prompt: "A ceramic vase with blue glaze"
  ↓
Trellis_SparseStructure_Sampler → sparse_latents
  ↓
... (same as image pipeline)
```

---

## Output Formats

| Format | Use Case | Export Node |
|--------|----------|-------------|
| **mesh** | General 3D model | `Trellis_Export_GLB` |
| **gaussian** | Point cloud rendering | `Trellis_Export_PLY` |
| **radiance** | NeRF-style view synthesis | *(decode only, no export)* |

---

## Parameters Guide

### Sparse Structure Sampler
- `cfg`: 5.0-10.0 (7.5 default) - Higher = more faithful to input
- `steps`: 8-20 (12 default) - More = better quality, slower

### SLAT Sampler
- `cfg`: 2.0-5.0 (3.0 default) - Refines geometry details
- `steps`: 8-20 (12 default) - More = smoother surface

### Decoder
- `texture_size`: 512/1024/2048 - Higher = better textures, more VRAM
- `mesh_simplify`: 0.9-0.98 - Reduce polygon count (0.95 = 5% decimation)

---

## Tips

**For Best Quality**:
- Remove backgrounds before processing
- Use `multidiffusion` for multi-view
- Increase steps to 20 for both samplers
- Use texture_size=2048 (if VRAM allows)

**For Speed**:
- Install flash-attn
- Reduce steps to 8
- Use `mode="fast"` in decoder
- Lower texture_size to 512

**For Pottery/Fragments**:
- Use multi-view with front+back images
- Enable `recenter` in preprocessor
- Try `stochastic` algorithm for variety
- Post-process with `MeshCraft_GeomOperations`

---

## Troubleshooting

**Out of Memory**:
- Reduce texture_size
- Use `mode="fast"` in decoder
- Process one view at a time

**Poor Quality**:
- Check image backgrounds (remove if needed)
- Increase CFG strength
- Add more steps
- Try multi-view instead of single image

**Model Download Fails**:
- Models auto-download from HuggingFace (~2-3GB)
- Manually download: `huggingface-cli download microsoft/TRELLIS-image-large`

---

## Example Workflows

See `workflows/` directory:
- `trellis-i2m.json` - Single image to 3D
- `trellis-t2m.json` - Text to 3D

---

## Advanced Features

### SLAT Visualization
```
Trellis_SLAT_Visualizer
├─ slat: (from SLAT_Sampler)
├─ filename: "debug_slat"
└─ colorscale: "Viridis"
  ↓
Outputs interactive 3D point cloud HTML
```

### Video Rendering
```
Trellis_Render_Video
├─ outputs: (from decoder)
├─ video_format: "mp4"
└─ fps: 30
  ↓
360° turntable animation
```

---

## Node Reference

**Loaders (3)**:
- `Load_DinoV2_Model` - Image encoder
- `Load_CLIP_Trellis` - Text encoder
- `Load_Trellis_Model` - Main pipeline

**Pipeline (9)**:
- `Trellis_Image_Preprocessor` - Clean/recenter images
- `Trellis_Image_Conditioning` - Encode images
- `Trellis_Text_Conditioning` - Encode text
- `Trellis_SparseStructure_Sampler` - Generate sparse voxels
- `Trellis_SLAT_Sampler` - Refine with structured latents
- `Trellis_SLAT_Decoder` - Decode to 3D
- `Trellis_SLAT_Visualizer` - Debug visualization
- `Trellis_Export_GLB` - Save mesh
- `Trellis_Export_PLY` - Save Gaussians
- `Trellis_Render_Video` - Create turntable

**Utilities (1)**:
- `Trellis_multiimage_loader` - Combine 2-3 images

---

**Total Nodes**: 17 (formerly 19, removed deprecated monolithic samplers)
