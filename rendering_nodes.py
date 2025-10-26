"""
ComfyUI-MeshCraft Rendering Nodes

Modular nodes for Hunyuan3D 2.1 multiview texture generation.
Provides transparent, composable pipeline for rendering conditioning maps
and generating PBR textures with optional geometry editing.
"""

import torch
import numpy as np
from typing import Tuple
from PIL import Image
import os

# ComfyUI imports
import folder_paths
import comfy.model_management as mm
import comfy.utils

# =============================================================================
# Lazy Import System - Heavy dependencies loaded on first use
# =============================================================================
_LAZY_IMPORTS = {}

def _lazy_import(module_name):
    """Cache and return heavy imports only when first needed"""
    if module_name not in _LAZY_IMPORTS:
        if module_name == "torch":
            import torch
            _LAZY_IMPORTS["torch"] = torch
        elif module_name == "trimesh":
            import trimesh as Trimesh
            _LAZY_IMPORTS["trimesh"] = Trimesh
        elif module_name == "MeshRender":
            from .hy3dpaint.DifferentiableRenderer.MeshRender import MeshRender
            _LAZY_IMPORTS["MeshRender"] = MeshRender
        elif module_name == "multiviewDiffusionNet":
            from .hy3dpaint.utils.multiview_utils import multiviewDiffusionNet
            _LAZY_IMPORTS["multiviewDiffusionNet"] = multiviewDiffusionNet
        elif module_name == "Hunyuan3DPaintConfig":
            from .hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig
            _LAZY_IMPORTS["Hunyuan3DPaintConfig"] = Hunyuan3DPaintConfig

    return _LAZY_IMPORTS.get(module_name)


# =============================================================================
# Utility Functions
# =============================================================================

def pil_images_to_tensor(images):
    """Convert list of PIL images to ComfyUI IMAGE tensor (NHWC, float32, 0-1)"""
    torch = _lazy_import("torch")
    np_images = [np.array(img).astype(np.float32) / 255.0 for img in images]
    tensor = torch.from_numpy(np.stack(np_images))
    return tensor


def tensor_to_pil_images(tensor):
    """Convert ComfyUI IMAGE tensor to list of PIL images"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # Handle batch or single image
    if len(tensor.shape) == 4:  # Batch (N, H, W, C)
        images = []
        for i in range(tensor.shape[0]):
            img_np = (tensor[i] * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))
        return images
    else:  # Single image (H, W, C)
        img_np = (tensor * 255).astype(np.uint8)
        return [Image.fromarray(img_np)]


# =============================================================================
# Node 1: Load Hunyuan Multiview Model
# =============================================================================

class LoadHunyuanMultiViewModel:
    """
    Load and cache Hunyuan3D 2.1 multiview diffusion model.

    This model generates PBR textures (albedo + metallic-roughness) from:
    - Reference image (what the object should look like)
    - Normal maps (geometric surface orientation)
    - Position maps (3D coordinates of surface points)

    Model is cached in memory for fast reuse across multiple generations.
    """

    # Class-level cache
    _cached_model = None
    _cached_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": ("STRING", {
                    "default": "tencent/Hunyuan3D-2.1",
                    "tooltip": "HuggingFace repository ID for the PaintPBR model"
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "attention_mode": (["sdpa", "sageattn", "flash", "xformers"], {
                    "default": "sdpa",
                    "tooltip": "Attention mechanism (sdpa=auto, sageattn=8-bit quantized, flash=FlashAttention, xformers=memory efficient)"
                }),
            },
        }

    RETURN_TYPES = ("MULTIVIEW_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MeshCraft/Rendering"

    def load_model(self, model_repo="tencent/Hunyuan3D-2.1", device="cuda", attention_mode="sdpa"):
        """
        Load Hunyuan3D multiview diffusion model.

        Args:
            model_repo: HuggingFace repository ID (e.g., "tencent/Hunyuan3D-2.1")
            device: Device to load model on ("cuda" or "cpu")
            attention_mode: Attention mechanism to use

        Returns:
            Tuple containing cached model instance
        """
        # Debug: Check what we actually received
        print(f"\n🎨 Loading Hunyuan3D Multiview Model...")
        print(f"   DEBUG: model_repo type={type(model_repo)}, value={repr(model_repo)}")
        print(f"   DEBUG: device type={type(device)}, value={repr(device)}")
        print(f"   DEBUG: attention_mode type={type(attention_mode)}, value={repr(attention_mode)}")

        # Validate inputs - catch if arguments are swapped
        if model_repo in ["cuda", "cpu"]:
            print(f"   ⚠️  WARNING: model_repo looks like a device! Swapping parameters...")
            model_repo, device = device, model_repo

        print(f"   Model: {model_repo}")
        print(f"   Device: {device}")
        print(f"   Attention: {attention_mode}")

        # Check cache (include model_repo and attention_mode in hash)
        config_hash = f"{model_repo}_{device}_{attention_mode}"
        if (LoadHunyuanMultiViewModel._cached_model is not None and
            LoadHunyuanMultiViewModel._cached_config_hash == config_hash):
            print("   ⚡ Using cached model (fast!)")
            return (LoadHunyuanMultiViewModel._cached_model,)

        print("   🔥 Loading model from HuggingFace (first time or settings changed)...")

        # Import dependencies
        multiviewDiffusionNet = _lazy_import("multiviewDiffusionNet")
        Hunyuan3DPaintConfig = _lazy_import("Hunyuan3DPaintConfig")

        # Create config with minimal parameters (model loading only)
        config = Hunyuan3DPaintConfig(
            resolution=512,  # Default, can be overridden in GenerateMultiviewPBR
            camera_azims=[0],  # Dummy values, not used in model loading
            camera_elevs=[0],
            view_weights=[1.0],
            ortho_scale=1.0,
            texture_size=1024
        )
        config.device = device
        config.multiview_pretrained_path = model_repo  # Use custom model repo

        # Load model
        model = multiviewDiffusionNet(config)

        # Set attention mode on the pipeline
        self._set_attention_mode(model.pipeline, attention_mode)

        # Cache it
        LoadHunyuanMultiViewModel._cached_model = model
        LoadHunyuanMultiViewModel._cached_config_hash = config_hash

        print("   ✅ Model loaded and cached!")
        print("=== End Model Loading ===\n")

        return (model,)

    def _set_attention_mode(self, pipeline, attention_mode):
        """Set attention mechanism on the pipeline's UNet."""
        print(f"   Setting attention mode: {attention_mode}")

        if attention_mode == "xformers":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("   ✅ XFormers attention enabled")
            except Exception as e:
                print(f"   ⚠️  Failed to enable XFormers: {e}")
                print(f"   💡 Install with: pip install xformers")

        elif attention_mode == "flash":
            # Flash Attention is used via SDPA backend when available
            # PyTorch will automatically use it if flash-attn is installed
            try:
                # Disable xformers if it was enabled
                if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'set_default_attn_processor'):
                    pipeline.unet.set_default_attn_processor()
                print("   ✅ Flash Attention enabled (via SDPA)")
                print("   💡 Ensure flash-attn is installed: pip install flash-attn")
            except Exception as e:
                print(f"   ⚠️  Error setting Flash Attention: {e}")

        elif attention_mode == "sageattn":
            try:
                # SageAttention works as a drop-in replacement for SDPA
                # The UNet's attention processors use F.scaled_dot_product_attention
                # which can be monkey-patched by sageattention if installed
                print("   ✅ SageAttention mode selected")
                print("   💡 Ensure sageattention is installed: pip install sageattention")
                print("   Note: SageAttention replaces SDPA at the PyTorch level")
            except Exception as e:
                print(f"   ⚠️  Error with SageAttention: {e}")

        elif attention_mode == "sdpa":
            # Default PyTorch SDPA (auto-selects best backend)
            try:
                if hasattr(pipeline, 'unet') and hasattr(pipeline.unet, 'set_default_attn_processor'):
                    pipeline.unet.set_default_attn_processor()
                print("   ✅ SDPA attention enabled (PyTorch auto-select)")
            except Exception as e:
                print(f"   ⚠️  Error setting SDPA: {e}")

        else:
            print(f"   ⚠️  Unknown attention mode: {attention_mode}, using default")


# =============================================================================
# Node 2: Render Conditioning Maps
# =============================================================================

class RenderConditioningMaps:
    """
    Render normal and position maps from mesh for Hunyuan3D conditioning.

    These maps encode the geometry of the mesh and are used as conditioning
    for the multiview diffusion model. They tell the model "what shape to
    texture" from each camera viewpoint.

    Normal maps: Surface orientation (which way each point faces)
    Position maps: 3D coordinates of each visible point

    Rendering uses exact Hunyuan3D 2.1 parameters for compatibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 256,
                    "tooltip": "Render resolution for each view"
                }),
                "ortho_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Orthographic camera scale"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("normal_maps", "position_maps")
    FUNCTION = "render"
    CATEGORY = "MeshCraft/Rendering"

    def render(self, mesh, camera_config, resolution, ortho_scale):
        """
        Render normal and position maps from mesh.

        Args:
            mesh: Input trimesh object
            camera_config: Camera configuration with azimuths/elevations
            resolution: Render resolution per view
            ortho_scale: Orthographic projection scale

        Returns:
            Tuple of (normal_maps, position_maps) as IMAGE batches
        """
        print(f"\n🎥 Rendering Conditioning Maps...")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Ortho scale: {ortho_scale}")

        # Import dependencies
        MeshRender = _lazy_import("MeshRender")
        torch = _lazy_import("torch")

        # Extract camera parameters
        azimuths = camera_config["selected_camera_azims"]
        elevations = camera_config["selected_camera_elevs"]
        num_views = len(azimuths)

        print(f"   Rendering {num_views} views...")

        # Create renderer with Hunyuan3D settings
        renderer = MeshRender(
            camera_distance=1.1,  # Hunyuan3D default
            camera_type="orth",
            default_resolution=resolution,
            texture_size=resolution,
            use_antialias=True,
            raster_mode="cr",
            shader_type="face",
            device=mm.get_torch_device(),
            ortho_scale=ortho_scale
        )

        # Load mesh into renderer
        renderer.load_mesh(mesh=mesh, scale_factor=1.15, auto_center=True)

        # Render normal maps
        normal_images = []
        for azim, elev in zip(azimuths, elevations):
            print(f"   Rendering normal (azim={azim}°, elev={elev}°)...")
            normal_map = renderer.render_normal(
                elev=elev,
                azim=azim,
                use_abs_coor=True,  # World-space normals (Hunyuan3D setting)
                normalize_rgb=True,
                return_type="pl"  # PIL Image
            )
            normal_images.append(normal_map)

        # Render position maps
        position_images = []
        for azim, elev in zip(azimuths, elevations):
            print(f"   Rendering position (azim={azim}°, elev={elev}°)...")
            position_map = renderer.render_position(
                elev=elev,
                azim=azim,
                return_type="pl"  # PIL Image
            )
            position_images.append(position_map)

        # Convert to ComfyUI IMAGE format
        normal_tensor = pil_images_to_tensor(normal_images)
        position_tensor = pil_images_to_tensor(position_images)

        print(f"   ✅ Rendered {num_views} views")
        print(f"   Normal maps shape: {normal_tensor.shape}")
        print(f"   Position maps shape: {position_tensor.shape}")
        print("=== End Conditioning Rendering ===\n")

        return (normal_tensor, position_tensor)


# =============================================================================
# Node 3: Render RGB Multiview Images (Blender-based)
# =============================================================================

class RenderRGBMultiview:
    """
    Render RGB images of mesh using Blender (matching Hunyuan3D-2.1 training pipeline).

    This uses Blender's Cycles/Eevee renderer to create high-quality multiview RGB images,
    exactly as described in Hunyuan3D-2.1 data preprocessing documentation.

    Based on the rendering pipeline from TRELLIS and Objaverse datasets.

    Uses:
    - Creating reference images matching Hunyuan3D training data
    - Generating synthetic training datasets
    - High-quality mesh visualization from multiple angles
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "resolution": ("INT", {
                    "default": 518,
                    "min": 64,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "Render resolution for each view (518 matches Hunyuan3D training)"
                }),
                "background_color": (["white", "transparent"], {
                    "default": "transparent",
                    "tooltip": "Background color for rendered images (Hunyuan3D uses transparent)"
                }),
                "engine": (["CYCLES", "BLENDER_EEVEE"], {
                    "default": "CYCLES",
                    "tooltip": "Rendering engine (Cycles=high quality, Eevee=fast)"
                }),
                "samples": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Number of samples for Cycles (Hunyuan3D uses 128)"
                }),
                "ortho_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Orthographic camera scale (Hunyuan3D uses 1.2)"
                }),
                "camera_distance": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Camera distance from object (Hunyuan3D uses 1.5)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_images",)
    FUNCTION = "render"
    CATEGORY = "MeshCraft/Rendering"

    def render(self, mesh, camera_config, resolution, background_color, engine, samples, ortho_scale, camera_distance):
        """
        Render RGB images using Blender (matching Hunyuan3D-2.1 training pipeline).

        Args:
            mesh: Input trimesh object
            camera_config: Camera configuration with azimuths/elevations
            resolution: Render resolution per view
            background_color: Background color ("white" or "transparent")
            engine: Rendering engine ("CYCLES" or "BLENDER_EEVEE")
            samples: Number of samples for Cycles rendering
            ortho_scale: Orthographic camera scale (Hunyuan3D uses 1.2)
            camera_distance: Camera distance from object (Hunyuan3D uses 1.5)

        Returns:
            Tuple containing rendered RGB images as IMAGE batch
        """
        print(f"\n🎨 Rendering RGB Multiview Images (Blender)...")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Background: {background_color}")
        print(f"   Engine: {engine}")
        print(f"   Samples: {samples}")
        print(f"   Ortho scale: {ortho_scale}")
        print(f"   Camera distance: {camera_distance}")

        import subprocess
        import tempfile
        import json
        from pathlib import Path
        import numpy as np
        from PIL import Image

        torch = _lazy_import("torch")

        # Extract camera parameters
        azimuths = camera_config["selected_camera_azims"]
        elevations = camera_config["selected_camera_elevs"]
        num_views = len(azimuths)

        print(f"   Rendering {num_views} views...")

        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save mesh to temporary file
            mesh_path = tmpdir / "mesh.obj"
            mesh.export(str(mesh_path))

            # Create camera config file
            camera_data = {
                "azimuths": azimuths,
                "elevations": elevations,
                "ortho_scale": ortho_scale,
                "camera_distance": camera_distance,
                "resolution": resolution,
                "background": background_color,
                "engine": engine,
                "samples": samples,
                "output_dir": str(tmpdir)
            }
            config_path = tmpdir / "render_config.json"
            with open(config_path, 'w') as f:
                json.dump(camera_data, f)

            # Use external Blender rendering script
            import os as os_module
            script_path = os_module.path.join(
                os_module.path.dirname(os_module.path.abspath(__file__)),
                "blender_render_script.py"
            )

            # Find Blender executable
            blender_path = self._find_blender()

            if blender_path is None:
                raise RuntimeError(
                    "Blender not found! Please install Blender 3.0+ or set BLENDER_PATH environment variable.\n"
                    "Download from: https://www.blender.org/download/"
                )

            # Run Blender in background mode
            print(f"   Running Blender: {blender_path}")
            cmd = [
                blender_path,
                "--background",
                "--python", str(script_path),
                "--",
                str(mesh_path),
                str(config_path)
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                # Always show output for debugging
                if result.stdout:
                    print(f"   Blender stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"   Blender stderr:\n{result.stderr}")

                if result.returncode != 0:
                    raise RuntimeError(f"Blender rendering failed with code {result.returncode}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender rendering timed out after 5 minutes")

            # Load rendered images
            rendered_images = []
            for i in range(num_views):
                img_path = tmpdir / f"render_{i:03d}.png"
                if not img_path.exists():
                    print(f"   ⚠️  Missing render {i}, creating blank")
                    blank = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
                    rendered_images.append(blank)
                else:
                    img = Image.open(img_path)

                    # Convert to RGB if needed
                    if img.mode == 'RGBA' and background_color == "white":
                        # Composite over white background
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    else:
                        img = img.convert('RGB')

                    rendered_images.append(np.array(img))

        # Convert to ComfyUI IMAGE format (NHWC, float32, 0-1)
        rendered_tensors = [torch.from_numpy(img.astype(np.float32) / 255.0) for img in rendered_images]
        output = torch.stack(rendered_tensors, dim=0)

        print(f"   ✅ Rendered {num_views} RGB views")
        print(f"   Output shape: {output.shape}")
        print("=== End RGB Multiview Rendering ===\n")

        return (output,)

    def _find_blender(self):
        """Find Blender executable."""
        import os
        import shutil

        # Check environment variable first
        blender_path = os.environ.get("BLENDER_PATH")
        if blender_path and os.path.exists(blender_path):
            return blender_path

        # Check common locations
        common_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "C:\\Program Files\\Blender Foundation\\Blender 4.1\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        # Try to find in PATH
        blender_path = shutil.which("blender")
        if blender_path:
            return blender_path

        return None


# =============================================================================
# Node 4: Generate Multiview PBR
# =============================================================================

class GenerateMultiviewPBR:
    """
    Generate PBR textures using Hunyuan3D multiview diffusion model.

    Takes a reference image and conditioning maps (normals + positions) and
    generates physically-based rendering (PBR) textures:
    - Albedo: Base color (diffuse color)
    - Metallic-Roughness (MR): Surface properties

    Key feature: Can accept EDITED conditioning maps to change geometry!
    For example, edit normal maps with Flux Kontext to make a handlebar curved,
    then generate textures that follow the edited geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MULTIVIEW_MODEL",),
                "reference_image": ("IMAGE",),
                "normal_maps": ("IMAGE",),
                "position_maps": ("IMAGE",),
                "steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of diffusion steps (more = better quality, slower)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "How closely to follow reference image"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 256,
                    "tooltip": "Output resolution (should match conditioning maps)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "mr")
    FUNCTION = "generate"
    CATEGORY = "MeshCraft/Rendering"

    def generate(self, model, reference_image, normal_maps, position_maps, steps, guidance_scale, seed, resolution):
        """
        Generate PBR textures using multiview diffusion.

        Args:
            model: Loaded multiview diffusion model
            reference_image: Reference image (what object should look like)
            normal_maps: Normal maps (can be edited!)
            position_maps: Position maps (can be edited!)
            steps: Number of diffusion steps
            guidance_scale: Guidance strength
            seed: Random seed
            resolution: Output resolution

        Returns:
            Tuple of (albedo_maps, mr_maps) as IMAGE batches
        """
        print(f"\n🎨 Generating Multiview PBR Textures...")
        print(f"   Steps: {steps}")
        print(f"   Guidance scale: {guidance_scale}")
        print(f"   Seed: {seed}")
        print(f"   Resolution: {resolution}")

        torch = _lazy_import("torch")

        # Convert reference image to PIL
        if isinstance(reference_image, torch.Tensor):
            ref_pil = tensor_to_pil_images(reference_image)
        else:
            ref_pil = [reference_image]

        # Convert conditioning maps to PIL
        normal_pil = tensor_to_pil_images(normal_maps)
        position_pil = tensor_to_pil_images(position_maps)

        num_views = len(normal_pil)
        print(f"   Processing {num_views} views...")

        # Prepare conditioning (normals + positions concatenated)
        conditioning = normal_pil + position_pil  # List of 2*num_views images

        # Clear CUDA cache before inference
        mm.soft_empty_cache()
        torch.cuda.empty_cache()

        # Run multiview diffusion
        print("   Running diffusion model...")
        output = model(
            ref_pil,  # Reference images
            conditioning,  # Normal + position maps
            prompt="high quality",  # Standard prompt
            custom_view_size=resolution,
            resize_input=True,
            num_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        # Extract albedo and MR
        albedo_pil = output["albedo"]  # List of PIL images
        mr_pil = output["mr"]  # List of PIL images

        # Convert to ComfyUI IMAGE format
        albedo_tensor = pil_images_to_tensor(albedo_pil)
        mr_tensor = pil_images_to_tensor(mr_pil)

        print(f"   ✅ Generated PBR textures")
        print(f"   Albedo shape: {albedo_tensor.shape}")
        print(f"   MR shape: {mr_tensor.shape}")
        print("=== End PBR Generation ===\n")

        return (albedo_tensor, mr_tensor)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "MeshCraft_LoadHunyuanMultiViewModel": LoadHunyuanMultiViewModel,
    "MeshCraft_RenderConditioningMaps": RenderConditioningMaps,
    "MeshCraft_RenderRGBMultiview": RenderRGBMultiview,
    "MeshCraft_GenerateMultiviewPBR": GenerateMultiviewPBR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshCraft_LoadHunyuanMultiViewModel": "Load Hunyuan Multiview Model",
    "MeshCraft_RenderConditioningMaps": "Render Conditioning Maps (Normals + Positions)",
    "MeshCraft_RenderRGBMultiview": "Render RGB Multiview Images",
    "MeshCraft_GenerateMultiviewPBR": "Generate Multiview PBR Textures",
}
