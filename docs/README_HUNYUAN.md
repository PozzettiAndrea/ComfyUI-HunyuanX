# Hunyuan 3D Cached Nodes for ComfyUI-MeshCraft

## Important: Prerequisites

These nodes require **ComfyUI-Hunyuan3d-2-1** to be installed first:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1
cd ComfyUI-Hunyuan3d-2-1
# Follow their installation instructions
```

## What This Provides

Optimized versions of Hunyuan 3D nodes with **model caching** for 5-10x faster reloading.

### Cached Nodes:
- `Hy3DMeshGenerator` - Caches pipeline between runs
- `Hy3D21VAELoader` - Caches VAE model
- `Hy3DMultiViewsGenerator` - Caches multiview model internally

### Usage

Each cached node has a `use_cache` parameter:
- `True` (default): Use cached models (faster)
- `False`: Always reload (useful for debugging)

## Attribution

Based on [ComfyUI-Hunyuan3d-2-1](https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1) by visualbruno.

**Modifications**:
- Added model caching for faster workflow execution
- Added cache toggle parameters
- Memory management improvements

**Powered by Tencent Hunyuan 3D 2.1** (required by license)

## License

Tencent Hunyuan 3D 2.1 Community License Agreement (see LICENSE_TENCENT_HUNYUAN)

Key terms:
- Free for research, education, and commercial use (<1M MAU)
- Must include attribution
- Cannot use to train competing AI models
- Geographic restrictions apply (check license)
