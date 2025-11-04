# Hunyuan 3D Integration

ComfyUI-MeshCraft includes **bundled** Hunyuan 3D 2.1 integration - no separate installation needed.

## Two Approaches

### Monolithic Pipeline (13 nodes)
All-in-one nodes for quick workflows. Model caching provides 5-10x speedup.

| Node | Purpose |
|------|---------|
| `Hy3DMeshGenerator` | Generate 3D from image (with caching) |
| `Hy3DMultiViewsGenerator` | Generate PBR multiview renders |
| `Hy3DBakeMultiViews` | Bake textures to UV |
| `Hy3DInPaint` | Fill texture holes |
| `Hy3D21CameraConfig` | Configure camera setup |
| `Hy3D21VAELoader` | Load VAE (with caching) |
| `Hy3D21VAEDecode` | Decode latents to mesh |
| `Hy3D21LoadImageWithTransparency` | Load RGBA image |
| `Hy3D21PostprocessMesh` | Clean up mesh |
| `Hy3D21ExportMesh` | Export to GLB/OBJ/STL/etc |
| `Hy3D21MeshUVWrap` | UV unwrap |
| `Hy3D21LoadMesh` | Load mesh from file |
| `Hy3D21ImageWithAlphaInput` | Combine image + mask |

### Modular Pipeline (9 nodes)
Memory-efficient approach with separate DiT/VAE loading (~2GB VRAM savings).

| Node | Purpose |
|------|---------|
| `PrepareImageForDINO` | Preprocess image |
| `LoadDinoModel` | Load DINO v2 encoder |
| `EncodeDINO` | Encode to embeddings |
| `LoadHunyuanDiT` | Load DiT only (saves VRAM) |
| `LoadHunyuanVAE` | Load VAE separately |
| `Hy3DGenerateLatents` | Generate 3D latents |
| `Hy3DDecodeLatents` | Decode to mesh |
| `Hy3DImageToLatents` | All-in-one convenience |
| `PreviewTrimesh` | Interactive 3D preview |

## Key Features

**Model Caching** (`use_cache` parameter):
- `True` (default): Cache models between runs (5-10x faster)
- `False`: Always reload (for debugging)

**Attention Modes**:
- `sdpa` - PyTorch scaled dot-product attention (default, compatible)
- `sageattn` - SageAttention (faster, requires installation)
- `flash` - FlashAttention (fastest, requires compilation)
- `xformers` - xFormers memory-efficient (good balance)

**FlashVDM Decoder**:
- Faster marching cubes decoding
- Optional toggle in VAE decode nodes

## Which to Use?

**Use Monolithic** when:
- Quick prototyping
- Standard workflows
- Don't need memory optimization

**Use Modular** when:
- Limited VRAM (<12GB)
- Want to cache DINO embeddings
- Need fine-grained control
- Experimenting with different DiT/VAE combinations

## Example Workflow

### Monolithic (Simple)
```
LoadImage → Hy3D21LoadImageWithTransparency → Hy3DMeshGenerator →
Hy3D21VAELoader → Hy3D21VAEDecode → Hy3D21ExportMesh
```

### Modular (Memory-Efficient)
```
LoadImage → PrepareImageForDINO → LoadDinoModel → EncodeDINO →
LoadHunyuanDiT → Hy3DGenerateLatents → LoadHunyuanVAE →
Hy3DDecodeLatents → ExportMesh
```

## Attribution

Based on Tencent's Hunyuan 3D 2.1 model.

**Modifications in MeshCraft**:
- Model caching system
- Lazy loading for fast startup
- Multiple attention backend support
- Modular pipeline option

## License

**Tencent Hunyuan 3D 2.1 Community License**

Key terms:
- ✅ Free for research and education
- ✅ Commercial use allowed (<1M monthly active users)
- ❌ Cannot train competing models
- ❌ Geographic restrictions apply (check full license)
- Required attribution: "Powered by Tencent Hunyuan 3D 2.1"

See `LICENSE_TENCENT_HUNYUAN` for full terms.

## Troubleshooting

**CUDA Extension Not Compiling**:
- Run `python install.py` to compile extensions
- Check CUDA Toolkit: `nvcc --version`
- Use precompiled wheels if available

**Out of Memory**:
- Use modular pipeline (saves ~2GB)
- Reduce texture_size parameter
- Enable FlashVDM for faster decoding
- Use `sage_attn` or `flash` attention modes

**Slow Generation**:
- Enable model caching (`use_cache=True`)
- Use `flash` or `sageattn` attention modes
- Reduce CFG steps

**Model Download Fails**:
- Models auto-download from HuggingFace (~4GB total)
- Check internet connection
- Manually download to `models/` directory if needed
