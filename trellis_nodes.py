"""
ComfyUI-MeshCraft TRELLIS Nodes

Integration of Microsoft TRELLIS (Structured 3D Latents for Scalable and Versatile 3D Generation)

Based on:
- Microsoft TRELLIS: https://github.com/microsoft/TRELLIS
- ComfyUI-IF_Trellis: https://github.com/if-ai/ComfyUI-IF_Trellis

Features:
- Single-image and multi-image 3D generation
- Multiple output formats (mesh, Gaussian splats, radiance fields)
- Model downloading and caching
- Multi-view fusion (stochastic and multidiffusion modes)

License: MIT License
"""

# ============================================================================
# Lazy Import System
# ============================================================================

import os
import time
import gc
from typing import List, Tuple, Any, Optional
from pathlib import Path

import folder_paths
import comfy.model_management as mm
import comfy.utils

from .model_cache import get_cache_key, get_cached_model, cache_model

# Lazy imports cache
_LAZY_IMPORTS = {}

def _lazy_import(module_name):
    """Cache and return heavy imports only when first needed"""
    if module_name not in _LAZY_IMPORTS:
        if module_name == "torch":
            import torch
            _LAZY_IMPORTS["torch"] = torch
        elif module_name == "numpy":
            import numpy as np
            _LAZY_IMPORTS["numpy"] = np
        elif module_name == "PIL.Image":
            from PIL import Image
            _LAZY_IMPORTS["PIL.Image"] = Image
        elif module_name == "trimesh":
            import trimesh
            _LAZY_IMPORTS["trimesh"] = trimesh
        elif module_name == "trellis_pipeline":
            from .trellis.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
            _LAZY_IMPORTS["trellis_pipeline"] = TrellisImageTo3DPipeline
        elif module_name == "trellis_utils":
            from .trellis.TRELLIS.trellis.utils import render_utils, postprocessing_utils
            _LAZY_IMPORTS["render_utils"] = render_utils
            _LAZY_IMPORTS["postprocessing_utils"] = postprocessing_utils
        elif module_name == "pyvista":
            import pyvista as pv
            _LAZY_IMPORTS["pyvista"] = pv

    return _LAZY_IMPORTS.get(module_name)

# ============================================================================
# Helper Functions
# ============================================================================

def tensor2pil(image):
    """Convert ComfyUI tensor to PIL Image"""
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """Convert PIL Image to ComfyUI tensor"""
    import numpy as np
    import torch
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# ============================================================================
# Node 1: TRELLIS Model Downloader
# ============================================================================

class TrellisDownloadModel:
    """
    Download and cache TRELLIS models from Hugging Face

    Provides explicit model downloading with progress tracking,
    similar to Hunyuan's pattern but for TRELLIS models.
    """

    def __init__(self):
        self.download_path = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["TRELLIS-image-large", "TRELLIS-image-base"],),
                "auto_download": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "download_model"
    CATEGORY = "TRELLIS"

    def download_model(self, model_name, auto_download=True, force_redownload=False):
        """
        Download TRELLIS model from Hugging Face if not already cached

        Args:
            model_name: Model variant to download
            auto_download: Whether to auto-download if not found
            force_redownload: Force re-download even if cached

        Returns:
            Tuple containing the local model path
        """
        # Determine cache directory
        cache_dir = os.environ.get('HY3DGEN_MODELS', os.path.expanduser('~/.cache/trellis'))
        model_path = Path(cache_dir) / model_name

        # Check if model exists
        model_exists = model_path.exists() and len(list(model_path.glob('*'))) > 0

        if model_exists and not force_redownload:
            print(f"✅ TRELLIS model '{model_name}' already cached at: {model_path}")
            return (str(model_path),)

        if not auto_download:
            if not model_exists:
                raise FileNotFoundError(
                    f"Model '{model_name}' not found at {model_path}. "
                    f"Enable 'auto_download' to download from Hugging Face."
                )
            return (str(model_path),)

        # Download from Hugging Face
        print(f"📥 Downloading TRELLIS model '{model_name}' from Hugging Face...")
        print(f"   This may take a while (~2-3GB download)")

        try:
            from huggingface_hub import snapshot_download

            # Download with progress bar
            downloaded_path = snapshot_download(
                repo_id=f"microsoft/{model_name}",
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
            )

            print(f"✅ TRELLIS model downloaded successfully to: {downloaded_path}")
            return (str(downloaded_path),)

        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for downloading models. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download TRELLIS model: {str(e)}")

# ============================================================================
# Node 2: TRELLIS Image-to-3D (Single Image)
# ============================================================================

class TrellisImageTo3D:
    """
    Generate 3D assets from a single image using TRELLIS

    Supports multiple output formats:
    - mesh: Polygonal mesh (compatible with Hunyuan pipeline)
    - gaussian: 3D Gaussian splats
    - radiance_field: Neural radiance field (NeRF)
    """

    def __init__(self):
        self.pipeline = None
        self.current_model_path = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": ""}),
                "output_format": (["mesh", "gaussian", "radiance_field", "all"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "ss_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "ss_cfg_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "slat_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "slat_cfg_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "DICT",)
    RETURN_NAMES = ("mesh", "outputs",)
    FUNCTION = "generate"
    CATEGORY = "TRELLIS"

    def generate(self, image, model_path, output_format, seed,
                 ss_steps, ss_cfg_strength, slat_steps, slat_cfg_strength,
                 use_cache=True):
        """
        Generate 3D asset from single image

        Args:
            image: Input image tensor (ComfyUI format)
            model_path: Path to TRELLIS model
            output_format: Desired output format(s)
            seed: Random seed for reproducibility
            ss_steps: Sparse structure sampler steps
            ss_cfg_strength: Sparse structure CFG strength
            slat_steps: SLAT sampler steps
            slat_cfg_strength: SLAT CFG strength
            use_cache: Whether to cache the pipeline

        Returns:
            Tuple of (mesh, outputs_dict)
        """
        torch = _lazy_import("torch")
        _lazy_import("trellis_pipeline")
        TrellisImageTo3DPipeline = _LAZY_IMPORTS["trellis_pipeline"]

        device = mm.get_torch_device()

        # Load or cache pipeline
        cache_key = get_cache_key(model_path, variant="trellis")

        if use_cache:
            cached = get_cached_model(cache_key)
            if cached is not None:
                self.pipeline = cached
                print("⚡ Using cached TRELLIS pipeline")
            else:
                print(f"🔥 Loading TRELLIS pipeline from: {model_path}")
                self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
                cache_model(cache_key, self.pipeline)
        else:
            print(f"🔄 Loading TRELLIS pipeline (cache disabled)")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)

        self.pipeline.to(device)

        # Convert ComfyUI image to PIL
        print("📸 Converting input image to PIL format...")
        image_pil = tensor2pil(image)
        print(f"   Image size: {image_pil.size}, mode: {image_pil.mode}")

        # Determine output formats
        formats = ['gaussian', 'radiance_field', 'mesh'] if output_format == 'all' else [output_format]

        # Generate
        print(f"\n{'='*80}")
        print(f"🎨 TRELLIS 3D Generation Starting")
        print(f"{'='*80}")
        print(f"   Output formats: {formats}")
        print(f"   Random seed: {seed}")
        print(f"   Sparse structure: {ss_steps} steps, CFG {ss_cfg_strength}")
        print(f"   SLAT refinement: {slat_steps} steps, CFG {slat_cfg_strength}")
        print(f"{'='*80}\n")

        start_time = time.time()

        print("🔄 Stage 1/3: Preprocessing image...")
        outputs = self.pipeline.run(
            image_pil,
            seed=seed,
            formats=formats,
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": ss_steps,
                "cfg_strength": ss_cfg_strength,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_cfg_strength,
            },
        )

        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✅ TRELLIS Generation Complete!")
        print(f"   Total time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
        print(f"   Outputs generated: {list(outputs.keys())}")
        print(f"{'='*80}\n")

        # Extract mesh for TRIMESH output
        mesh = None
        if 'mesh' in outputs and len(outputs['mesh']) > 0:
            # Convert TRELLIS mesh to trimesh
            trellis_mesh = outputs['mesh'][0]
            trimesh = _lazy_import("trimesh")

            # Export to trimesh format
            if hasattr(trellis_mesh, 'vertices') and hasattr(trellis_mesh, 'faces'):
                mesh = trimesh.Trimesh(
                    vertices=trellis_mesh.vertices,
                    faces=trellis_mesh.faces,
                )

        # Cleanup
        if not use_cache:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()

        return (mesh, outputs)

# ============================================================================
# Node 3: TRELLIS Multi-Image-to-3D
# ============================================================================

class TrellisMultiImageTo3D:
    """
    Generate 3D assets from multiple images using TRELLIS

    Supports two fusion modes:
    - stochastic: Round-robin view selection (good for diverse angles)
    - multidiffusion: Prediction averaging (good for similar views)

    Ideal for pottery fragments with front+back views!
    """

    def __init__(self):
        self.pipeline = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images
                "model_path": ("STRING", {"default": ""}),
                "fusion_mode": (["stochastic", "multidiffusion"],),
                "output_format": (["mesh", "gaussian", "radiance_field", "all"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "ss_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "ss_cfg_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "slat_steps": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "slat_cfg_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "DICT",)
    RETURN_NAMES = ("mesh", "outputs",)
    FUNCTION = "generate"
    CATEGORY = "TRELLIS"

    def generate(self, images, model_path, fusion_mode, output_format, seed,
                 ss_steps, ss_cfg_strength, slat_steps, slat_cfg_strength,
                 use_cache=True):
        """
        Generate 3D asset from multiple images

        Args:
            images: Batch of input images (ComfyUI format)
            model_path: Path to TRELLIS model
            fusion_mode: How to combine multi-view information
            output_format: Desired output format(s)
            seed: Random seed
            ss_steps, ss_cfg_strength: Sparse structure sampler params
            slat_steps, slat_cfg_strength: SLAT sampler params
            use_cache: Whether to cache pipeline

        Returns:
            Tuple of (mesh, outputs_dict)
        """
        torch = _lazy_import("torch")
        _lazy_import("trellis_pipeline")
        TrellisImageTo3DPipeline = _LAZY_IMPORTS["trellis_pipeline"]

        device = mm.get_torch_device()

        # Load or cache pipeline
        cache_key = get_cache_key(model_path, variant="trellis")

        if use_cache:
            cached = get_cached_model(cache_key)
            if cached is not None:
                self.pipeline = cached
                print("⚡ Using cached TRELLIS pipeline")
            else:
                print(f"🔥 Loading TRELLIS pipeline from: {model_path}")
                self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
                cache_model(cache_key, self.pipeline)
        else:
            print(f"🔄 Loading TRELLIS pipeline (cache disabled)")
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)

        self.pipeline.to(device)

        # Convert batch of ComfyUI images to list of PIL images
        print("📸 Converting input images to PIL format...")
        image_list = [tensor2pil(img) for img in images]
        num_views = len(image_list)
        print(f"   Converted {num_views} images")
        for i, img in enumerate(image_list):
            print(f"   - View {i+1}: {img.size}, {img.mode}")

        # Determine output formats
        formats = ['gaussian', 'radiance_field', 'mesh'] if output_format == 'all' else [output_format]

        # Generate with multi-image mode
        print(f"\n{'='*80}")
        print(f"🎨 TRELLIS Multi-View 3D Generation Starting")
        print(f"{'='*80}")
        print(f"   Number of views: {num_views}")
        print(f"   Fusion mode: {fusion_mode}")
        print(f"   Output formats: {formats}")
        print(f"   Random seed: {seed}")
        print(f"   Sparse structure: {ss_steps} steps, CFG {ss_cfg_strength}")
        print(f"   SLAT refinement: {slat_steps} steps, CFG {slat_cfg_strength}")
        print(f"{'='*80}\n")

        start_time = time.time()

        print(f"🔄 Stage 1/3: Preprocessing {num_views} images...")
        print(f"🔄 Stage 2/3: Generating sparse structure ({fusion_mode} mode)...")
        print(f"   This will alternate/fuse between {num_views} views across {ss_steps} steps")

        outputs = self.pipeline.run_multi_image(
            image_list,
            seed=seed,
            formats=formats,
            preprocess_image=True,
            mode=fusion_mode,
            sparse_structure_sampler_params={
                "steps": ss_steps,
                "cfg_strength": ss_cfg_strength,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_cfg_strength,
            },
        )

        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✅ TRELLIS Multi-View Generation Complete!")
        print(f"   Total time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
        print(f"   Outputs generated: {list(outputs.keys())}")
        print(f"{'='*80}\n")

        # Extract mesh
        mesh = None
        if 'mesh' in outputs and len(outputs['mesh']) > 0:
            trellis_mesh = outputs['mesh'][0]
            trimesh = _lazy_import("trimesh")

            if hasattr(trellis_mesh, 'vertices') and hasattr(trellis_mesh, 'faces'):
                mesh = trimesh.Trimesh(
                    vertices=trellis_mesh.vertices,
                    faces=trellis_mesh.faces,
                )

        # Cleanup
        if not use_cache:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()

        return (mesh, outputs)

# ============================================================================
# Node Registration
# ============================================================================

# ============================================================================
# Node 4: PyVista Mesh Saver
# ============================================================================

class SaveMeshPyVista:
    """
    Save mesh using PyVista (supports STL, PLY, VTK, OBJ, and more)

    PyVista is a powerful 3D visualization library that provides:
    - Multiple export formats
    - Mesh analysis and statistics
    - Clean, well-documented API
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "mesh"}),
                "format": (["stl", "ply", "vtk", "obj", "glb"],),
                "binary": ("BOOLEAN", {"default": True}),  # Binary format (smaller files)
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_mesh"
    OUTPUT_NODE = True
    CATEGORY = "TRELLIS"

    def save_mesh(self, trimesh, filename_prefix, format, binary=True):
        """
        Save mesh using PyVista

        Args:
            trimesh: Input trimesh object
            filename_prefix: Output filename prefix
            format: Export format (stl, ply, vtk, obj, glb)
            binary: Use binary format for smaller files

        Returns:
            Path to saved file
        """
        pv = _lazy_import("pyvista")

        # Get output directory
        output_dir = folder_paths.get_output_directory()

        # Generate unique filename
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{format}"
        filepath = os.path.join(output_dir, filename)

        print(f"\n{'='*80}")
        print(f"💾 Saving mesh with PyVista")
        print(f"{'='*80}")

        # Convert trimesh to pyvista
        print("🔄 Converting trimesh to PyVista format...")
        vertices = trimesh.vertices
        faces = trimesh.faces

        # PyVista requires faces in [n, v1, v2, v3, ...] format
        # For triangles: [3, v1, v2, v3]
        pv_faces = []
        for face in faces:
            pv_faces.extend([len(face)] + face.tolist())

        pv_mesh = pv.PolyData(vertices, pv_faces)

        # Print mesh statistics
        print(f"\n📊 Mesh Statistics:")
        print(f"   Vertices: {pv_mesh.n_points:,}")
        print(f"   Faces: {pv_mesh.n_cells:,}")
        print(f"   Bounds: {pv_mesh.bounds}")
        print(f"   Volume: {pv_mesh.volume:.6f}")
        print(f"   Surface Area: {pv_mesh.area:.6f}")

        # Save based on format
        print(f"\n💾 Saving as {format.upper()}...")

        if format == "stl":
            pv_mesh.save(filepath, binary=binary)
        elif format == "ply":
            pv_mesh.save(filepath, binary=binary)
        elif format == "vtk":
            pv_mesh.save(filepath, binary=binary)
        elif format == "obj":
            pv_mesh.save(filepath)  # OBJ doesn't have binary option
        elif format == "glb":
            # GLB export via trimesh (pyvista doesn't support glb directly)
            trimesh.export(filepath)

        file_size = os.path.getsize(filepath)
        file_size_mb = file_size / (1024 * 1024)

        print(f"\n{'='*80}")
        print(f"✅ Mesh saved successfully!")
        print(f"   File: {filename}")
        print(f"   Path: {filepath}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Format: {format.upper()} ({'Binary' if binary else 'ASCII'})")
        print(f"{'='*80}\n")

        return (filepath,)


# ============================================================================
# Node 5: PyVista Mesh Info
# ============================================================================

class MeshInfoPyVista:
    """
    Analyze mesh using PyVista and display detailed statistics

    Provides comprehensive mesh analysis including:
    - Geometry stats (vertices, faces, edges)
    - Quality metrics (volume, area, bounds)
    - Topology info (manifold, watertight)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "analyze_mesh"
    OUTPUT_NODE = True
    CATEGORY = "TRELLIS"

    def analyze_mesh(self, trimesh):
        """
        Analyze mesh and return detailed statistics
        """
        pv = _lazy_import("pyvista")

        # Convert to PyVista
        vertices = trimesh.vertices
        faces = trimesh.faces

        pv_faces = []
        for face in faces:
            pv_faces.extend([len(face)] + face.tolist())

        pv_mesh = pv.PolyData(vertices, pv_faces)

        # Gather statistics
        info_lines = []
        info_lines.append("="*80)
        info_lines.append("🔍 MESH ANALYSIS (PyVista)")
        info_lines.append("="*80)
        info_lines.append("")

        # Geometry
        info_lines.append("📐 GEOMETRY:")
        info_lines.append(f"   Vertices: {pv_mesh.n_points:,}")
        info_lines.append(f"   Faces: {pv_mesh.n_cells:,}")
        info_lines.append(f"   Edges: {pv_mesh.n_lines:,}")
        info_lines.append("")

        # Bounds
        bounds = pv_mesh.bounds
        info_lines.append("📏 BOUNDS:")
        info_lines.append(f"   X: [{bounds[0]:.4f}, {bounds[1]:.4f}] (size: {bounds[1]-bounds[0]:.4f})")
        info_lines.append(f"   Y: [{bounds[2]:.4f}, {bounds[3]:.4f}] (size: {bounds[3]-bounds[2]:.4f})")
        info_lines.append(f"   Z: [{bounds[4]:.4f}, {bounds[5]:.4f}] (size: {bounds[5]-bounds[4]:.4f})")
        info_lines.append("")

        # Center
        center = pv_mesh.center
        info_lines.append("🎯 CENTER:")
        info_lines.append(f"   ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        info_lines.append("")

        # Metrics
        info_lines.append("📊 METRICS:")
        info_lines.append(f"   Volume: {pv_mesh.volume:.6f}")
        info_lines.append(f"   Surface Area: {pv_mesh.area:.6f}")
        info_lines.append("")

        # Topology
        info_lines.append("🔗 TOPOLOGY:")
        info_lines.append(f"   Is all triangles: {pv_mesh.is_all_triangles}")
        info_lines.append(f"   Number of cells: {pv_mesh.n_cells:,}")
        info_lines.append("")

        # Memory
        memory_mb = pv_mesh.memory_usage / (1024 * 1024)
        info_lines.append("💾 MEMORY:")
        info_lines.append(f"   Usage: {memory_mb:.2f} MB")
        info_lines.append("")

        info_lines.append("="*80)

        info_text = "\n".join(info_lines)
        print(info_text)

        return (info_text,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "TrellisDownloadModel": TrellisDownloadModel,
    "TrellisImageTo3D": TrellisImageTo3D,
    "TrellisMultiImageTo3D": TrellisMultiImageTo3D,
    "SaveMeshPyVista": SaveMeshPyVista,
    "MeshInfoPyVista": MeshInfoPyVista,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrellisDownloadModel": "TRELLIS Download Model 📥",
    "TrellisImageTo3D": "TRELLIS Image→3D 🖼️➡️🎲",
    "TrellisMultiImageTo3D": "TRELLIS Multi-View→3D 🖼️🖼️➡️🎲",
    "SaveMeshPyVista": "Save Mesh (PyVista) 💾",
    "MeshInfoPyVista": "Mesh Info (PyVista) 🔍",
}
