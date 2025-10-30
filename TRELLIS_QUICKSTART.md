# TRELLIS Quick Start Guide

Microsoft TRELLIS integration for ComfyUI-MeshCraft - Multi-view 3D generation with sparse latents.

## Overview

TRELLIS (Structured 3D Latents) is a cutting-edge 3D generation model from Microsoft Research that:

- ✅ Supports **single-image** and **multi-image** 3D generation
- ✅ Generates multiple output formats (mesh, Gaussian splats, radiance fields)
- ✅ Handles **partial/fragmented objects** better than dense voxel approaches
- ✅ Ideal for **archaeological pottery fragments** (front+back reconstruction)
- ✅ Uses **tuning-free multi-view fusion** (no retraining needed!)

**Paper**: [Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506) (CVPR 2025 Spotlight)

---

## Installation

### 1. Basic Dependencies (Auto-installed)

The prestartup script automatically installs:
- `imageio` + `ffmpeg` support
- `einops` for tensor operations

### 2. Optional Performance Boost

For 10-20% faster inference:

```bash
# Option A: Flash Attention (recommended for RTX 4090/5090)
pip install flash-attn --no-build-isolation

# Option B: xformers (alternative)
pip install xformers
```

### 3. Advanced Dependencies (Manual)

For full TRELLIS functionality:

```bash
# Sparse convolution (version-specific to your CUDA version)
pip install spconv-cu118  # For CUDA 11.8

# 3D operations library
pip install kaolin

# Differentiable rendering
pip install nvdiffrast
```

---

## Quick Start: Basic Workflow

### Step 1: Download Model

```
TrellisDownloadModel 📥
├─ model_name: "TRELLIS-image-large"  # 2-3GB download
├─ auto_download: true
└─ → model_path (STRING)
```

### Step 2: Generate 3D from Single Image

```
Load Image
  ↓
TrellisImageTo3D 🖼️➡️🎲
├─ image: IMAGE
├─ model_path: (from Step 1)
├─ output_format: "mesh"
├─ seed: 42
├─ ss_steps: 12             # Sparse structure steps
├─ ss_cfg_strength: 7.5
├─ slat_steps: 12           # SLAT refinement steps
├─ slat_cfg_strength: 3.0
└─ → mesh (TRIMESH), outputs (DICT)
```

### Step 3: Save or Post-Process

```
mesh → Hy3D21ExportMesh → Save GLB
```

---

## Advanced: Multi-View Pottery Reconstruction

Perfect for **front+back pottery fragments**!

### Workflow

```
Load Image (front view)
Load Image (back view)
  ↓ (batch images)
TrellisMultiImageTo3D 🖼️🖼️➡️🎲
├─ images: IMAGE (batch of 2-8 images)
├─ model_path: (from download node)
├─ fusion_mode: "stochastic" or "multidiffusion"
│   ├─ stochastic: Round-robin view selection (good for opposite angles like front+back)
│   └─ multidiffusion: Prediction averaging (good for similar viewpoints)
├─ output_format: "mesh"
├─ seed: 42
├─ ss_steps: 12
├─ ss_cfg_strength: 7.5
├─ slat_steps: 12
├─ slat_cfg_strength: 3.0
└─ → mesh (TRIMESH), outputs (DICT)
```

### Fusion Modes Explained

#### Stochastic Mode (Recommended for Front+Back)

```
Steps: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
Views: [F,  B,  F,  B,  F,  B,  F,  B,  F,  B,  F,  B]
       Front alternates with Back
```

**Pros:**
- Better for **opposing viewpoints** (e.g., 0° and 180°)
- Faster inference (one view at a time)
- Works well with 2-8 images

**Cons:**
- May miss features if `num_images > num_steps`

#### Multidiffusion Mode

```
At each step:
  pred_front = model(front_image)
  pred_back = model(back_image)
  final_pred = (pred_front + pred_back) / 2
```

**Pros:**
- Better for **similar viewpoints** (e.g., 0°, 45°, 90°)
- Smoother blending

**Cons:**
- Slower (N× forward passes per step)
- May conflict with opposite angles

---

## Hybrid Workflow: TRELLIS + Hunyuan

**Best of both worlds**: TRELLIS geometry + Hunyuan textures

```
Load Image (pottery fragment)
  ↓
TrellisImageTo3D (generate base geometry)
  ↓ mesh
Hy3D21PostprocessMesh (clean topology)
  ↓ mesh
Hy3DMultiViewsGenerator (add high-quality PBR textures)
  ├─ mesh: (from TRELLIS)
  ├─ image: (original reference image)
  └─ → textured mesh
Hy3D21ExportMesh (save GLB with textures)
```

**Why this works:**
- TRELLIS excels at **geometry reconstruction** for partial objects
- Hunyuan excels at **PBR texture generation** (albedo, metallic, roughness)
- Combined result has accurate shape + photorealistic materials

---

## Tips for Pottery Fragments

### Input Image Preparation

1. **Preprocessing**: Remove background (use `rembg` or manual masking)
2. **Lighting**: Uniform, diffuse lighting works best
3. **Resolution**: 518×518 px (TRELLIS will auto-resize)
4. **Format**: PNG with alpha channel preferred

### Parameter Tuning

| Scenario | ss_steps | ss_cfg_strength | slat_steps | slat_cfg_strength |
|----------|----------|-----------------|------------|-------------------|
| **Fast preview** | 8 | 5.0 | 8 | 2.0 |
| **Balanced (default)** | 12 | 7.5 | 12 | 3.0 |
| **High quality** | 20 | 10.0 | 20 | 4.0 |
| **Pottery fragments** | 15 | 8.0 | 15 | 3.5 |

**Rules of thumb:**
- Higher `ss_steps` = better coarse geometry
- Higher `slat_steps` = finer surface details
- Higher `cfg_strength` = stronger adherence to input image

### Multi-View Tips

**For front+back pottery:**

```yaml
fusion_mode: "stochastic"  # Better for opposite angles
ss_steps: 18               # Extra steps for complex geometry
slat_steps: 18             # Extra refinement for details
seed: 42                   # Reproducibility
```

**If adding side views:**

```yaml
images: [front, left, back, right]  # 90° increments
fusion_mode: "multidiffusion"       # Smoother blending
ss_steps: 12
slat_steps: 12
```

---

## Output Formats

### Mesh (Default)

```
output_format: "mesh"
→ TRIMESH object (vertices + faces)
```

**Compatible with:**
- Hunyuan texture generation
- MeshLab post-processing
- Blender import
- GLB/OBJ export

### Gaussian Splats

```
output_format: "gaussian"
→ 3D Gaussian splat representation
```

**Use cases:**
- Real-time rendering
- Neural rendering pipelines
- Export to `.ply` or `.splat` formats

### Radiance Field (NeRF)

```
output_format: "radiance_field"
→ Neural radiance field representation
```

**Use cases:**
- Novel view synthesis
- Volume rendering
- Research applications

### All Formats

```
output_format: "all"
→ outputs dict with all three formats
```

---

## Troubleshooting

### "TRELLIS package not found"

**Cause**: TRELLIS submodule not initialized

**Solution**:
```bash
cd ComfyUI/custom_nodes/ComfyUI-MeshCraft
git submodule update --init --recursive
```

### "spconv not installed" warnings

**Cause**: Sparse convolution library not installed

**Solution**:
```bash
# Install for your CUDA version
pip install spconv-cu118  # CUDA 11.8
pip install spconv-cu121  # CUDA 12.1
```

### Out of memory (CUDA OOM)

**Solutions**:
1. Reduce steps: `ss_steps=8, slat_steps=8`
2. Use fewer images in multi-view mode
3. Disable model caching: `use_cache=false`
4. Clear GPU memory: Restart ComfyUI

### Poor geometry quality

**Fixes**:
1. Increase steps: `ss_steps=20, slat_steps=20`
2. Adjust CFG strength: Try `ss_cfg_strength=10.0`
3. Try different seeds (TRELLIS can be sensitive)
4. Preprocess input: remove background, adjust lighting

### Multi-view artifacts

**Fixes**:
1. Switch fusion mode (`stochastic` ↔ `multidiffusion`)
2. Increase steps for smoother blending
3. Ensure consistent lighting across views
4. Remove backgrounds from all input images

---

## Performance Benchmarks

**Hardware**: NVIDIA RTX 4090 (24GB VRAM)

| Mode | Resolution | Steps (SS/SLAT) | Time | VRAM |
|------|------------|-----------------|------|------|
| Single image | 518×518 | 12/12 | ~15s | 10GB |
| Multi-view (2 images) | 518×518 | 12/12 | ~25s | 12GB |
| Multi-view (4 images) | 518×518 | 12/12 | ~40s | 14GB |
| High quality | 518×518 | 20/20 | ~30s | 12GB |

**With Flash Attention**: 10-20% faster

---

## Examples

### Example 1: Single Fragment

```
Input: Single pottery shard image
Node: TrellisImageTo3D
Parameters: Default (12/12 steps)
Output: Clean 3D mesh ready for texturing
Time: ~15 seconds
```

### Example 2: Front+Back Reconstruction

```
Inputs: Front view + Back view of pottery fragment
Node: TrellisMultiImageTo3D
Parameters: fusion_mode="stochastic", 18/18 steps
Output: Complete 3D reconstruction capturing both sides
Time: ~30 seconds
```

### Example 3: Hybrid Pipeline

```
Step 1: TRELLIS generates geometry (15s)
Step 2: Hunyuan Paint adds PBR textures (45s)
Total: ~60 seconds for production-ready GLB
```

---

## References

- **Paper**: [Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506)
- **GitHub**: https://github.com/microsoft/TRELLIS
- **Project Page**: https://trellis3d.github.io/
- **Model**: `microsoft/TRELLIS-image-large` on Hugging Face

---

## License

TRELLIS is released under the MIT License by Microsoft Research.

ComfyUI-MeshCraft TRELLIS integration: MIT License

---

**Questions?** Open an issue at https://github.com/YOUR_USERNAME/ComfyUI-MeshCraft/issues
