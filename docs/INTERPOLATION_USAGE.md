# MeshCraft Interpolation Nodes Usage Guide

## Overview

The MeshCraft interpolation nodes allow you to create smooth transitions between two images using either:
1. **VAE Latent Interpolation** - For 2D image morphing
2. **Hunyuan3D Latent Interpolation** - For 3D mesh morphing
3. **Dual Interpolation** - Run both for comparison

All nodes use **SLERP (Spherical Linear Interpolation)** by default, which is the best practice for latent space interpolation in generative models.

## Why SLERP?

- ✅ Preserves vector magnitude in latent space
- ✅ Produces smoother, more natural transitions
- ✅ Avoids statistical issues that cause gray/distorted images with LERP
- ✅ Recommended by the Stable Diffusion community

## Available Nodes

### 1. DINO Embedding Interpolation ⭐ RECOMMENDED!

**Purpose:** Interpolate between semantic feature embeddings with precise control

**Workflow:**
```
Image 1 ──► PrepareImageForDINO ──► EncodeDINO ──► Embedding 1 ──┐
                                                                    ├─► DINO Embedding Interpolation ──► Single Embedding
Image 2 ──► PrepareImageForDINO ──► EncodeDINO ──► Embedding 2 ──┘      (factor: 0.0 to 1.0 slider)
                                                                                    ↓
                                                                         Hy3DGenerateLatents ──► Mesh
```

**Why DINO Interpolation?**
- ✅ Works in **semantic feature space** (high-level concepts)
- ✅ Better than raw VAE for semantic transitions (e.g., "plant disappears")
- ✅ Can guide 3D generation with interpolated features
- ✅ More meaningful transitions than pixel-space interpolation
- ✅ **Slider control** - manually adjust from 0.0 to 1.0 for precise control

**Inputs:**
- `embedding1`: First DINO embedding (from `EncodeDINO` node)
- `embedding2`: Second DINO embedding (from `EncodeDINO` node)
- `factor`: Interpolation factor **slider** (0.0 to 1.0)
  - **0.0** = pure embedding1 (pot + plant)
  - **0.5** = halfway between
  - **1.0** = pure embedding2 (just pot)
- `interpolation_method`: "slerp" (default) or "lerp"

**Output:**
- Single `DINO_EMBEDDING` at the specified factor

**How to Use:**
- Connect both embeddings
- Adjust the factor slider to explore the semantic space
- Feed output to `Hy3DGenerateLatents` to create a mesh
- Want animation? Queue multiple times with different factors!

**Use Cases:**
- Semantic morphing (plant → no plant in feature space)
- Exploring what the model "understands" about the transition
- Creating smoother 3D transitions based on semantic understanding

---

### 3. VAE Latent Interpolation

**Purpose:** Interpolate between images in 2D latent space

**Workflow:**
```
Image 1 ──┐
          ├─► VAE Latent Interpolation ──► Single Image
Image 2 ──┤    (factor: 0.0 to 1.0 slider)
VAE ──────┘
```

**Inputs:**
- `image1`: Starting image
- `image2`: Ending image
- `vae`: VAE model (use ComfyUI's "Load VAE" node)
- `factor`: Interpolation factor **slider** (0.0 to 1.0)
  - **0.0** = pure image1
  - **0.5** = halfway between
  - **1.0** = pure image2
- `interpolation_method`: "slerp" (default) or "lerp"

**Output:**
- Single `IMAGE` at the specified factor

**How to Use:**
- Load both images and a VAE
- Connect to the interpolation node
- Adjust the factor slider to see transitions
- Want animation? Queue multiple times with different factors!

**Use Cases:**
- Creating morph animations between two photos
- Exploring the visual latent space
- Generating intermediate images for video transitions

---

### 2. Hunyuan3D Latent Interpolation

**Purpose:** Interpolate in 3D latent space with precise control

**Workflow:**
```
Image 1 (pot+plant) ──► Hy3DGenerateLatents ──► Latent 1 ──┐
                                                             ├─► Hunyuan3D Latent Interpolation ──► Single Latent
Image 2 (just pot)  ──► Hy3DGenerateLatents ──► Latent 2 ──┘      (factor: 0.0 to 1.0 slider)
                                                                              ↓
                                                                    Hy3DLatentDecoder ──► Mesh
```

**Inputs:**
- `latent1`: Hunyuan3D latent from first image (from `Hy3DGenerateLatents`)
- `latent2`: Hunyuan3D latent from second image
- `factor`: Interpolation factor **slider** (0.0 to 1.0)
  - **0.0** = pure latent1
  - **0.5** = halfway between
  - **1.0** = pure latent2
- `interpolation_method`: "slerp" (default) or "lerp"

**Output:**
- Single `HY3DLATENT` at the specified factor

**How to Use:**
- Generate latents for both images first
- Connect both to the interpolation node
- Adjust the factor slider to explore transitions
- Decode the output to get your mesh
- Want animation? Queue multiple times with different factors!

**Use Cases:**
- Creating "plant growth" animations in 3D
- Morphing between different 3D objects
- Exploring Hunyuan3D's latent space

---

## Example: Pot with Plant → Just Pot

### Setup

1. **Prepare your images:**
   - Image A: Pot with plant
   - Image B: Just the pot

2. **Load them in ComfyUI:**
   - Use "Load Image" nodes for both

### Workflow A: 2D VAE Interpolation

```
[Load Image: pot_with_plant.jpg]  ┐
                                   ├─► [VAE Latent Interpolation]  ──► [Save Image]
[Load Image: just_pot.jpg]         │      factor: 0.5 (slider)
                                   │      method: slerp
[Load VAE]  ───────────────────────┘
```

This will generate a single image at factor 0.5 (halfway between both images). Adjust the slider to 0.0, 0.25, 0.5, 0.75, 1.0 and queue 5 times to create an animation sequence.

### Workflow B: DINO Embedding Interpolation (Recommended for Semantic Transitions!)

```
[Load Image: pot_with_plant.jpg] ──► [PrepareImageForDINO] ──► [EncodeDINO] ──► embedding1 ──┐
                                                                                               │
[Load Image: just_pot.jpg] ──────► [PrepareImageForDINO] ──► [EncodeDINO] ──► embedding2 ──┤
                                                                                               │
                                                                                               ▼
                                                                          [DINO Embedding Interpolation]
                                                                                  factor: 0.3 (slider)
                                                                                  method: slerp
                                                                                               │
                                                                                               ▼
                                                                              [Hy3DGenerateLatents]
                                                                                               │
                                                                                               ▼
                                                                              [Hy3DLatentDecoder]
                                                                                               │
                                                                                               ▼
                                                                                 Single 3D mesh
```

This interpolates in **semantic space** and generates a single 3D mesh at factor 0.3. Adjust the slider and queue multiple times to create an animation sequence showing the plant gradually disappearing based on what the model understands about the scene.

### Workflow C: 3D Hunyuan3D Latent Interpolation

```
[Load Image: pot_with_plant.jpg] ──► [Hy3DGenerateLatents] ──► latent1 ──┐
                                                                            │
                                                                            ├─► [Hunyuan3D Latent Interpolation]  ──► [Hy3DLatentDecoder] ──► [Preview Trimesh]
[Load Image: just_pot.jpg] ──────► [Hy3DGenerateLatents] ──► latent2 ──┘      factor: 0.7 (slider)
                                                                                 method: slerp
```

This will generate a single 3D mesh at factor 0.7 (closer to the "just pot" latent). Adjust the slider and queue multiple times to create an animation sequence showing the plant gradually disappearing in 3D latent space.

---

## Tips & Best Practices

### Creating Animations

Since each node returns a **single output** at the specified factor:
- **Manual approach:** Adjust slider, queue, repeat (factor: 0.0, 0.1, 0.2, ..., 1.0)
- **Recommended frames:** 10-20 for smooth animations
- **Hunyuan3D:** 5-10 frames (each mesh takes time to generate)

### SLERP vs LERP

- **Always use SLERP** for latent space interpolation
- Only use LERP for experimentation/comparison
- SLERP is mathematically correct for high-dimensional spaces

### Performance

- **VAE interpolation** is fast (seconds)
- **Hunyuan3D interpolation** is slow (minutes per mesh)
- Consider using fewer steps for Hunyuan3D

### Quality

- Use high-quality input images with clean backgrounds
- Ensure both images have similar lighting and scale
- For 3D: Pre-process images with background removal

---

## Troubleshooting

### "Latent shapes don't match" error

**Problem:** The two latents have different dimensions

**Solution:**
- Ensure both images are the same resolution
- Use the same Hunyuan3D model settings for both images
- Check that VAE is the same for both images

### Gray or distorted intermediate images

**Problem:** Using LERP instead of SLERP

**Solution:**
- Switch `interpolation_method` to "slerp"

### Out of memory errors

**Problem:** Too many interpolation steps for Hunyuan3D

**Solution:**
- Reduce `steps` to 5 or fewer
- Process latents in smaller batches
- Use a GPU with more VRAM

---

## Advanced: Batch Processing

The interpolation nodes output batches, which you can:

1. **Save as animation:**
   - Use ComfyUI's "Save Animation" nodes
   - Export as GIF or video (MP4)

2. **Process individually:**
   - Use "Latent From Batch" to extract single frames
   - Apply per-frame post-processing

3. **Create 3D animation:**
   - Export mesh batch as OBJ/GLB sequence
   - Import into Blender for animation

---

## Technical Details

### SLERP Implementation

Based on Andrej Karpathy's reference implementation:
https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355

### Latent Space Info

- **VAE latents:** Compressed image representations (typically 512x512 → 64x64x4)
- **Hunyuan3D latents:** Compressed 3D shape representations using ShapeVAE

### References

- [Stable Diffusion Interpolation Cookbook](https://huggingface.co/learn/cookbook/stable_diffusion_interpolation)
- [SLERP for Stable Diffusion](https://dev.to/ramgendeploy/exploiting-latent-vectors-in-stable-diffusion-interpolation-and-parameters-tuning-j3d)
- [Hunyuan3D Paper](https://arxiv.org/abs/2411.02293)

---

## Questions?

- Check ComfyUI console for debug output
- Interpolation progress is printed during processing
- Report issues to MeshCraft GitHub repository
