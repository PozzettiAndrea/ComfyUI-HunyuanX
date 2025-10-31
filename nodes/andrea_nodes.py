"""
Andrea's Modular Hunyuan3D Nodes

Splits the monolithic mesh generator into modular, reusable components:
- PrepareImageForDINO: Image preprocessing (recenter, resize to 518x518)
- EncodeDINO: DINO v2 encoding (cacheable embeddings)
- Hy3DGenerateLatents: Latent generation with exposed advanced parameters
- Hy3DImageToLatents: Convenience all-in-one node
"""

import os
import gc
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import trimesh as Trimesh

import folder_paths
import comfy.model_management as mm

from ..utils.model_cache import get_cache_key, get_cached_model, cache_model


# =============================================================================
# Lazy Imports (from hunyuan_nodes.py pattern)
# =============================================================================

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
        elif module_name == "hunyuan_pipeline":
            from ..lib.hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            _LAZY_IMPORTS["hunyuan_pipeline"] = Hunyuan3DDiTFlowMatchingPipeline
        elif module_name == "comfy_utils":
            from comfy.utils import load_torch_file
            _LAZY_IMPORTS["load_torch_file"] = load_torch_file
        elif module_name == "trimesh":
            import trimesh as Trimesh
            _LAZY_IMPORTS["trimesh"] = Trimesh
    return _LAZY_IMPORTS.get(module_name)


# =============================================================================
# Utility Functions
# =============================================================================

def tensor2pil(image):
    """Convert ComfyUI IMAGE tensor to PIL Image"""
    np = _lazy_import("numpy")
    _lazy_import("PIL.Image")
    Image = _LAZY_IMPORTS["PIL.Image"]
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    """Convert PIL Image to ComfyUI IMAGE tensor"""
    np = _lazy_import("numpy")
    torch = _lazy_import("torch")
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def recenter_image(image_np, border_ratio=0.15):
    """
    Recenter an image to leave border space (from preprocessors.py).

    Args:
        image_np: RGBA numpy array [H, W, 4] uint8
        border_ratio: Border ratio (0.15 = 15% border, 85% object)

    Returns:
        Tuple of (RGB image, mask) both as uint8 numpy arrays
    """
    np = _lazy_import("numpy")

    if image_np.shape[-1] == 4:
        mask = image_np[..., 3]
    else:
        mask = np.ones_like(image_np[..., 0:1]) * 255
        image_np = np.concatenate([image_np, mask], axis=-1)
        mask = mask[..., 0]

    H, W, C = image_np.shape
    size = max(H, W)
    result = np.zeros((size, size, C), dtype=np.uint8)

    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        raise ValueError('Input image is empty (no non-zero pixels in mask)')

    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min

    if h == 0 or w == 0:
        raise ValueError('Input image bounding box has zero size')

    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2

    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
        image_np[x_min:x_max, y_min:y_max], (w2, h2),
        interpolation=cv2.INTER_AREA
    )

    # Composite on white background
    bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255
    mask_alpha = result[..., 3:].astype(np.float32) / 255
    result = result[..., :3] * mask_alpha + bg * (1 - mask_alpha)
    result = result.clip(0, 255).astype(np.uint8)
    mask = (mask_alpha * 255).clip(0, 255).astype(np.uint8)

    return result, mask


# =============================================================================
# Node 1: PrepareImageForDINO
# =============================================================================

class PrepareImageForDINO:
    """
    Preprocess image for DINO encoding.

    Steps:
    1. Convert ComfyUI IMAGE to PIL/numpy
    2. Recenter object with border
    3. Resize to 518x518 (DINO's native size)
    4. Normalize to [-1, 1]

    This replaces ImageProcessorV2 with explicit control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_ratio": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Border ratio - 0.15 means object takes 85% of canvas"
                }),
                "size": ("INT", {
                    "default": 518,
                    "min": 128,
                    "max": 1024,
                    "step": 2,
                    "tooltip": "Target size (should be 518 for DINO)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed_image",)
    FUNCTION = "prepare"
    CATEGORY = "Hunyuan3D21/Preprocessing"

    def prepare(self, image, border_ratio, size):
        torch = _lazy_import("torch")
        np = _lazy_import("numpy")
        _lazy_import("PIL.Image")
        Image = _LAZY_IMPORTS["PIL.Image"]

        # Convert ComfyUI IMAGE tensor to PIL
        pil_image = tensor2pil(image)

        # Convert to RGBA
        pil_image = pil_image.convert("RGBA")
        image_np = np.asarray(pil_image)

        # Recenter with border
        centered_rgb, centered_mask = recenter_image(image_np, border_ratio)

        # Resize to target size
        centered_rgb = cv2.resize(centered_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        centered_mask = cv2.resize(centered_mask, (size, size), interpolation=cv2.INTER_NEAREST)

        # Normalize to [-1, 1] (DINO's expected range)
        image_normalized = centered_rgb.astype(np.float32) / 255.0
        image_normalized = image_normalized * 2 - 1

        # Convert to tensor [1, H, W, 3]
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)

        return (image_tensor,)


# =============================================================================
# Node 2a: LoadDinoModel
# =============================================================================

class LoadDinoModel:
    """
    Load DINO v2 model from HuggingFace.

    Supports all DINO v2 variants: Small, Base, Large, Giant.
    The loaded model is cached and can be reused across workflows.
    """

    # Class-level cache variables (persist across node instances)
    _cached_conditioner = None
    _cached_model_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (["small", "base", "large", "giant"], {
                    "default": "large",
                    "tooltip": "DINO v2 model size (Hunyuan3D 2.1 uses Large)"
                }),
                "keep_in_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM between runs (recommended for faster execution)"
                }),
            }
        }

    RETURN_TYPES = ("DINO_MODEL",)
    RETURN_NAMES = ("dino_model",)
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D21/Models"

    def load(self, model_size, keep_in_memory=True):
        torch = _lazy_import("torch")

        device = mm.get_torch_device()

        # Map size to HuggingFace model ID
        model_map = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large",
            "giant": "facebook/dinov2-giant",
        }

        model_id = model_map[model_size]

        # Clear cache if keep_in_memory is False
        if not keep_in_memory and LoadDinoModel._cached_conditioner is not None:
            print(f"🗑️ Clearing DINO model from cache")
            del LoadDinoModel._cached_conditioner
            LoadDinoModel._cached_conditioner = None
            LoadDinoModel._cached_model_id = None

        # Check if already loaded
        if LoadDinoModel._cached_conditioner is not None and LoadDinoModel._cached_model_id == model_id:
            print(f"⚡ Using cached DINO model ({model_id})")
            return (LoadDinoModel._cached_conditioner,)

        print(f"📥 Downloading DINO v2 from HuggingFace: {model_id}")

        try:
            from transformers import Dinov2Model

            # Download DINO model
            dino_model = Dinov2Model.from_pretrained(model_id)

            # Wrap in conditioner structure compatible with Hunyuan pipeline
            class DinoConditioner:
                def __init__(self, model):
                    self.model = model
                    self.main_image_encoder = self

                def to(self, device):
                    self.model.to(device)
                    return self

                def eval(self):
                    self.model.eval()
                    return self

                def __call__(self, image, **kwargs):
                    # Return in Hunyuan's expected format: {'main': tensor}
                    outputs = self.model(image)
                    return {'main': outputs.last_hidden_state}

            LoadDinoModel._cached_conditioner = DinoConditioner(dino_model)
            print(f"✅ Loaded DINO v2 {model_size}: {model_id}")

            # Print model info
            total_params = sum(p.numel() for p in dino_model.parameters())
            print(f"   Parameters: {total_params:,}")

        except Exception as e:
            raise ValueError(f"Failed to load DINO from HuggingFace: {e}")

        LoadDinoModel._cached_conditioner.to(device).to(torch.float16)
        LoadDinoModel._cached_conditioner.eval()
        LoadDinoModel._cached_model_id = model_id

        if keep_in_memory:
            print(f"💾 DINO model cached in memory for faster reloads")

        return (LoadDinoModel._cached_conditioner,)


# =============================================================================
# Node 2b: EncodeDINO (Updated to use LoadDinoModel)
# =============================================================================

class EncodeDINO:
    """
    Encode image using DINO v2 model.

    Accepts a DINO model from LoadDinoModel node.
    Returns DINO embeddings that can be cached and reused for multiple
    latent generations with different seeds/parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dino_model": ("DINO_MODEL", {"tooltip": "DINO model from LoadDinoModel"}),
                "image": ("IMAGE", {"tooltip": "Preprocessed image (should be 518x518, range [-1,1])"}),
            }
        }

    RETURN_TYPES = ("DINO_EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "encode"
    CATEGORY = "Hunyuan3D21/Encoding"

    def encode(self, dino_model, image):
        torch = _lazy_import("torch")

        device = mm.get_torch_device()

        print("\n=== EncodeDINO Debug ===")
        print(f"Input image shape: {image.shape} {image.dtype}")

        # Ensure model is on correct device
        dino_model.to(device)

        # Convert image tensor to correct format
        # ComfyUI IMAGE is [B, H, W, C], need [B, C, H, W]
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
            print(f"Permuted image shape: {image.shape}")

        image = image.to(device, dtype=dino_model.model.dtype)
        print(f"Final image shape: {image.shape} {image.dtype} device={image.device}")

        # Encode with DINO
        with torch.no_grad():
            cond = dino_model(image=image)

        print(f"Output conditioning type: {type(cond)}")
        if isinstance(cond, dict):
            print(f"Conditioning keys: {list(cond.keys())}")
            for key, value in cond.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} {value.dtype}")
                elif isinstance(value, dict):
                    print(f"  {key}: nested dict with keys {list(value.keys())}")

        print("=== End EncodeDINO Debug ===\n")

        gc.collect()

        return (cond,)


# =============================================================================
# Node 3: Hy3DGenerateLatents
# =============================================================================

class Hy3DGenerateLatents:
    """
    Generate 3D latents using Flow Matching DiT.

    Accepts DiT model from LoadHunyuanDiT and DINO embeddings.
    Runs the diffusion process to generate 3D latents.
    Exposes advanced parameters like sigmas and eta.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit_model": ("HY3D_DIT_MODEL", {"tooltip": "DiT model from LoadHunyuanDiT"}),
                "dino_embedding": ("DINO_EMBEDDING", {"tooltip": "DINO embeddings from EncodeDINO node"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "sigmas_str": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated sigmas (e.g., '0.0,0.5,1.0') or leave empty for auto"
                }),
            }
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D21/Generation"

    def generate(self, dit_model, dino_embedding, steps, guidance_scale, seed, sigmas_str=""):
        torch = _lazy_import("torch")
        import numpy as np
        from diffusers.utils.torch_utils import randn_tensor
        from comfy.utils import ProgressBar

        device = mm.get_torch_device()
        seed = seed % (2**32)

        print("\n=== Hy3DGenerateLatents Debug ===")
        print(f"Input dino_embedding type: {type(dino_embedding)}")

        # Unpack DiT model and scheduler
        model, scheduler = dit_model

        model.to(device)
        model.eval()

        print(f"Device: {device}")
        print(f"Steps: {steps}, Guidance scale: {guidance_scale}, Seed: {seed}")

        # Parse sigmas if provided
        if sigmas_str and sigmas_str.strip():
            try:
                sigmas = [float(s.strip()) for s in sigmas_str.split(',')]
                sigmas = np.array(sigmas)
                print(f"Using custom sigmas: {sigmas}")
            except Exception as e:
                print(f"Warning: Failed to parse sigmas '{sigmas_str}': {e}")
                print("Falling back to auto sigmas")
                sigmas = None
        else:
            sigmas = None

        # Prepare conditioning
        batch_size = 1

        # Check if classifier-free guidance is enabled
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(model, 'guidance_embed') and
            model.guidance_embed is True
        )

        print(f"\nDo classifier-free guidance: {do_classifier_free_guidance}")
        print(f"Input dino_embedding structure:")

        def print_dict_shapes(d, indent=0):
            for key, value in d.items():
                if isinstance(value, torch.Tensor):
                    print(f"{'  ' * indent}{key}: {value.shape} {value.dtype} device={value.device}")
                elif isinstance(value, dict):
                    print(f"{'  ' * indent}{key}: dict")
                    print_dict_shapes(value, indent + 1)
                else:
                    print(f"{'  ' * indent}{key}: {type(value)}")

        print_dict_shapes(dino_embedding)

        # Prepare conditioning (duplicate for CFG if needed)
        if do_classifier_free_guidance:
            # Create unconditional embedding (zeros)
            def create_uncond(cond_dict):
                uncond = {}
                for key, value in cond_dict.items():
                    if isinstance(value, torch.Tensor):
                        uncond[key] = torch.zeros_like(value)
                    elif isinstance(value, dict):
                        uncond[key] = create_uncond(value)
                    else:
                        uncond[key] = value
                return uncond

            uncond_embedding = create_uncond(dino_embedding)

            # Concatenate conditional and unconditional
            def cat_cond(cond1, cond2):
                result = {}
                for key in cond1.keys():
                    if isinstance(cond1[key], torch.Tensor):
                        result[key] = torch.cat([cond1[key], cond2[key]], dim=0)
                    elif isinstance(cond1[key], dict):
                        result[key] = cat_cond(cond1[key], cond2[key])
                    else:
                        result[key] = cond1[key]
                return result

            cond = cat_cond(dino_embedding, uncond_embedding)
        else:
            cond = dino_embedding

        print(f"\nFinal conditioning structure:")
        print_dict_shapes(cond)

        # Prepare timesteps using scheduler's method
        if sigmas is not None:
            # Custom sigmas provided - pass to scheduler
            scheduler.set_timesteps(sigmas=sigmas, device=device)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
            print(f"\nUsing custom sigmas: {len(sigmas)} values")
            print(f"Scheduler timesteps: {timesteps.shape}, num_steps: {num_inference_steps}")
            print(f"Timestep range: [{timesteps.min().item()}, {timesteps.max().item()}]")
        else:
            # Use scheduler's set_timesteps to properly initialize internal registry
            scheduler.set_timesteps(num_inference_steps=steps, device=device)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
            print(f"\nScheduler timesteps: {timesteps.shape}, num_steps: {num_inference_steps}")
            print(f"Timestep range: [{timesteps.min().item()}, {timesteps.max().item()}]")

        # Prepare latents
        # Get latent shape from model config
        latent_shape = (batch_size, 4096, 64)  # From config: num_latents=4096, in_channels=64
        generator = torch.manual_seed(seed)

        # Get dtype from model parameters
        dtype = next(model.parameters()).dtype

        print(f"\nLatent shape: {latent_shape}, dtype: {dtype}")

        latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)

        print(f"Initial latents: {latents.shape} {latents.dtype}")

        # Diffusion loop
        print(f"\nStarting diffusion loop...")
        pbar = ProgressBar(num_inference_steps)
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if i == 0:
                    print(f"\nStep {i}: t={t.item():.4f}")

                # Expand latents if doing classifier-free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                if i == 0:
                    print(f"  latent_model_input: {latent_model_input.shape}")

                # Normalize timestep to [0, 1]
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / scheduler.config.num_train_timesteps

                if i == 0:
                    print(f"  timestep: {timestep.shape} = {timestep}")

                # Predict noise
                guidance = None
                if hasattr(model, 'guidance_embed') and model.guidance_embed is True:
                    guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
                    if i == 0:
                        print(f"  guidance: {guidance}")

                if i == 0:
                    print(f"  Calling model with:")
                    print(f"    latent_model_input: {latent_model_input.shape}")
                    print(f"    timestep: {timestep.shape}")
                    print(f"    cond keys: {list(cond.keys())}")

                noise_pred = model(latent_model_input, timestep, cond, guidance=guidance)

                if i == 0:
                    print(f"  noise_pred: {noise_pred.shape}")

                # Apply classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    if i == 0:
                        print(f"  noise_pred_cond: {noise_pred_cond.shape}, noise_pred_uncond: {noise_pred_uncond.shape}")
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if i == 0:
                    print(f"  Final noise_pred: {noise_pred.shape}")

                # Compute previous sample
                outputs = scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if i == 0:
                    print(f"  Updated latents: {latents.shape}")

                # Update progress bar
                pbar.update_absolute(i + 1, num_inference_steps)

        print(f"\nDiffusion complete!")
        print(f"Final latents: {latents.shape} {latents.dtype}")
        print("=== End Debug ===\n")

        gc.collect()

        return (latents,)


# =============================================================================
# Node 4: Hy3DImageToLatents (Convenience)
# =============================================================================

class Hy3DImageToLatents:
    """
    All-in-one convenience node: Image → Latents

    For simple workflows that don't need embedding caching.
    Combines PrepareImageForDINO → EncodeDINO → Hy3DGenerateLatents.
    """

    def __init__(self):
        self.pipeline = None
        self.current_model_path = None
        self.current_attention_mode = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"),),
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "attention_mode": (["sdpa", "sageattn", "flash", "xformers"], {"default": "sdpa"}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "border_ratio": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5}),
                "skip_preprocessing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, assumes image is already preprocessed to 518x518"
                }),
            }
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D21/Convenience"

    def generate(self, model, image, steps, guidance_scale, seed, attention_mode,
                 use_cache, border_ratio=0.15, skip_preprocessing=False):
        torch = _lazy_import("torch")
        _lazy_import("hunyuan_pipeline")
        Hunyuan3DDiTFlowMatchingPipeline = _LAZY_IMPORTS["hunyuan_pipeline"]

        device = mm.get_torch_device()
        seed = seed % (2**32)
        model_path = folder_paths.get_full_path("diffusion_models", model)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        meshcraft_root = os.path.dirname(script_directory)  # Go up from nodes/ to MeshCraft root

        # Load pipeline if needed
        should_load = (
            not use_cache or
            self.pipeline is None or
            self.current_model_path != model_path or
            self.current_attention_mode != attention_mode
        )

        if should_load:
            if not use_cache:
                print(f"🔄 Loading pipeline: {model} (cache disabled)")
            else:
                print(f"🔥 Loading pipeline: {model} (first time or settings changed)")

            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                config_path=os.path.join(meshcraft_root, 'lib', 'hunyuan_configs', 'dit_config_2_1.yaml'),
                ckpt_path=model_path,
                attention_mode=attention_mode
            )

            if use_cache:
                self.current_model_path = model_path
                self.current_attention_mode = attention_mode
        else:
            print("⚡ Using cached pipeline (fast!)")

        self.pipeline.to(device)

        # Preprocess if needed
        if not skip_preprocessing:
            # Use PrepareImageForDINO logic
            prep_node = PrepareImageForDINO()
            image, = prep_node.prepare(image, border_ratio, 518)

        # Convert to PIL for pipeline
        pil_image = tensor2pil(image)

        # Call pipeline normally (it will do preprocessing internally)
        latents = self.pipeline(
            image=pil_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed),
            output_type="latent",
        )

        gc.collect()

        return (latents,)


# =============================================================================
# Node 5: LoadHunyuanDiT
# =============================================================================

class LoadHunyuanDiT:
    """
    Load only the Hunyuan DiT diffusion model + scheduler.

    Saves ~2GB of VRAM by not loading the VAE.
    Use with Hy3DGenerateLatents for latent generation,
    then load VAE separately with LoadHunyuanVAE for decoding.

    Auto-downloads from HuggingFace if not found locally.
    """

    # Class-level cache variables (persist across node instances)
    _cached_dit_model = None
    _cached_scheduler = None
    _cached_model_path = None
    _cached_attention_mode = None

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of local models
        local_models = folder_paths.get_filename_list("diffusion_models")

        # Add HuggingFace download options
        hf_models = [
            "tencent/Hunyuan3D-2 (fp16)",
            "tencent/Hunyuan3D-2.1 (fp16)",
        ]

        # Combine: local models first, then HF options
        all_models = local_models + hf_models

        return {
            "required": {
                "model": (all_models,),
                "attention_mode": (["sdpa", "sageattn", "flash", "xformers"], {"default": "sdpa"}),
                "use_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("HY3D_DIT_MODEL",)
    RETURN_NAMES = ("dit_model",)
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D21/Models"

    def load(self, model, attention_mode, use_cache):
        torch = _lazy_import("torch")
        _lazy_import("comfy_utils")
        load_torch_file = _LAZY_IMPORTS["load_torch_file"]

        device = mm.get_torch_device()

        # Check if it's a HuggingFace model ID
        if model.startswith("tencent/"):
            # Parse HF model ID
            model_id = model.split(" ")[0]  # Remove (fp16) suffix

            # Map to local filename
            if "2.1" in model:
                local_filename = "hunyuan3d-dit-v2-1-fp16.ckpt"
            else:
                local_filename = "hunyuan3d-dit-v2-0-fp16.ckpt"

            # Check if already downloaded
            diffusion_models_dir = folder_paths.get_folder_paths("diffusion_models")[0]
            model_path = os.path.join(diffusion_models_dir, local_filename)

            if not os.path.exists(model_path):
                print(f"📥 Downloading {model_id} from HuggingFace...")
                self._download_from_hf(model_id, model_path, "diffusion_models")
            else:
                print(f"✓ Found local model: {local_filename}")
        else:
            # Local model
            model_path = folder_paths.get_full_path("diffusion_models", model)

        cache_key = f"{model_path}_{attention_mode}"

        # Check if already loaded
        if use_cache and LoadHunyuanDiT._cached_dit_model is not None and LoadHunyuanDiT._cached_model_path == model_path and LoadHunyuanDiT._cached_attention_mode == attention_mode:
            print(f"⚡ Using cached DiT model")
            return ((LoadHunyuanDiT._cached_dit_model, LoadHunyuanDiT._cached_scheduler),)

        print(f"📥 Loading Hunyuan DiT model from: {os.path.basename(model_path)}")

        script_directory = os.path.dirname(os.path.abspath(__file__))
        meshcraft_root = os.path.dirname(script_directory)  # Go up from nodes/ to MeshCraft root
        config_path = os.path.join(meshcraft_root, 'lib', 'hunyuan_configs', 'dit_config_2_1.yaml')

        import yaml
        from ..lib.hy3dshape.hy3dshape.pipelines import instantiate_from_config

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set attention mode
        config['model']['params']['attention_mode'] = attention_mode

        # Load checkpoint
        ckpt = load_torch_file(model_path)

        # Load ONLY DiT model and scheduler
        dit_model = instantiate_from_config(config['model'])
        dit_model.load_state_dict(ckpt['model'])

        scheduler = instantiate_from_config(config['scheduler'])

        dit_model.to(device).to(torch.float16)
        dit_model.eval()

        print(f"✅ Loaded DiT model (skipped VAE for memory savings)")

        # Print model info
        total_params = sum(p.numel() for p in dit_model.parameters())
        print(f"   DiT parameters: {total_params:,}")

        if use_cache:
            LoadHunyuanDiT._cached_dit_model = dit_model
            LoadHunyuanDiT._cached_scheduler = scheduler
            LoadHunyuanDiT._cached_model_path = model_path
            LoadHunyuanDiT._cached_attention_mode = attention_mode

        return ((dit_model, scheduler),)

    def _download_from_hf(self, model_id, local_path, subfolder):
        """Download model from HuggingFace"""
        try:
            from huggingface_hub import hf_hub_download
            import shutil

            print(f"Downloading from {model_id}...")

            # Map model ID to the correct repo path
            if "2.1" in model_id:
                repo_file = "hunyuan3d-dit-v2-1/model.fp16.ckpt"
            else:
                repo_file = "hunyuan3d-dit-v2-0/model.fp16.ckpt"

            # Download to cache
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=repo_file,
                repo_type="model"
            )

            # Copy to ComfyUI models folder
            shutil.copy(downloaded_path, local_path)
            print(f"✅ Downloaded to {local_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to download {model_id}: {e}")


# =============================================================================
# Node 6: LoadHunyuanVAE
# =============================================================================

class LoadHunyuanVAE:
    """
    Load only the Hunyuan VAE decoder.

    Use after generating latents to decode them into 3D meshes.
    Can be loaded/unloaded independently from DiT to save VRAM.

    Auto-downloads from HuggingFace if not found locally.
    """

    # Class-level cache variables (persist across node instances)
    _cached_vae = None
    _cached_model_path = None

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of local VAE models
        local_models = folder_paths.get_filename_list("vae")

        # Add HuggingFace download options
        hf_models = [
            "tencent/Hunyuan3D-2 VAE (fp16)",
            "tencent/Hunyuan3D-2.1 VAE (fp16)",
        ]

        # Combine: local models first, then HF options
        all_models = local_models + hf_models

        return {
            "required": {
                "model": (all_models,),
                "use_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("HY3DVAE",)  # Compatible with original Hunyuan nodes
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D21/Models"

    def load(self, model, use_cache):
        torch = _lazy_import("torch")
        _lazy_import("comfy_utils")
        load_torch_file = _LAZY_IMPORTS["load_torch_file"]

        device = mm.get_torch_device()

        # Check if it's a HuggingFace model ID
        if model.startswith("tencent/"):
            # Parse HF model ID
            model_id = model.split(" ")[0]  # Remove "VAE (fp16)" suffix

            # Map to local filename
            if "2.1" in model:
                local_filename = "Hunyuan3D-vae-v2-1-fp16.ckpt"
            else:
                local_filename = "Hunyuan3D-vae-v2-0-fp16.ckpt"

            # Check if already downloaded
            vae_dir = folder_paths.get_folder_paths("vae")[0]
            model_path = os.path.join(vae_dir, local_filename)

            if not os.path.exists(model_path):
                print(f"📥 Downloading {model_id} VAE from HuggingFace...")
                self._download_from_hf(model_id, model_path)
            else:
                print(f"✓ Found local VAE: {local_filename}")
        else:
            # Local model
            model_path = folder_paths.get_full_path("vae", model)

        # Check if already loaded
        if use_cache and LoadHunyuanVAE._cached_vae is not None and LoadHunyuanVAE._cached_model_path == model_path:
            print(f"⚡ Using cached VAE")
            return (LoadHunyuanVAE._cached_vae,)

        print(f"📥 Loading Hunyuan VAE from: {os.path.basename(model_path)}")

        script_directory = os.path.dirname(os.path.abspath(__file__))
        meshcraft_root = os.path.dirname(script_directory)  # Go up from nodes/ to MeshCraft root
        config_path = os.path.join(meshcraft_root, 'lib', 'hunyuan_configs', 'dit_config_2_1.yaml')

        import yaml
        from ..lib.hy3dshape.hy3dshape.pipelines import instantiate_from_config

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load checkpoint
        ckpt = load_torch_file(model_path)

        # Load ONLY VAE
        vae = instantiate_from_config(config['vae'])

        # Handle both standalone VAE and full checkpoint
        if 'vae' in ckpt:
            # Full checkpoint with DiT + VAE
            vae.load_state_dict(ckpt['vae'], strict=False)
        else:
            # Standalone VAE checkpoint
            vae.load_state_dict(ckpt, strict=False)

        vae.to(device).to(torch.float16)
        vae.eval()

        print(f"✅ Loaded VAE (skipped DiT for memory savings)")

        # Print model info
        total_params = sum(p.numel() for p in vae.parameters())
        print(f"   VAE parameters: {total_params:,}")

        if use_cache:
            LoadHunyuanVAE._cached_vae = vae
            LoadHunyuanVAE._cached_model_path = model_path

        return (vae,)

    def _download_from_hf(self, model_id, local_path):
        """Download VAE from HuggingFace"""
        try:
            from huggingface_hub import hf_hub_download
            import shutil

            print(f"Downloading VAE from {model_id}...")

            # Map model ID to the correct repo path
            if "2.1" in model_id:
                repo_file = "hunyuan3d-vae-v2-1/model.fp16.ckpt"
            else:
                repo_file = "hunyuan3d-vae-v2-0/model.fp16.ckpt"

            # Download to cache
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=repo_file,
                repo_type="model"
            )

            # Copy to ComfyUI models folder
            shutil.copy(downloaded_path, local_path)
            print(f"✅ Downloaded VAE to {local_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to download VAE from {model_id}: {e}")


# =============================================================================
# Node 7: Hy3DDecodeLatents
# =============================================================================

class Hy3DDecodeLatents:
    """
    Decode Hunyuan 3D latents to 3D mesh using VAE.

    Accepts latents from Hy3DGenerateLatents and VAE from LoadHunyuanVAE.
    Performs marching cubes surface extraction to create the final mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("HY3DVAE",),  # Compatible with original Hunyuan nodes
                "latents": ("HY3DLATENT",),
                "box_v": ("FLOAT", {"default": 1.01, "min": 0.5, "max": 2.0, "step": 0.01}),
                "octree_resolution": ("INT", {"default": 384, "min": 64, "max": 1024, "step": 8}),
                "mc_level": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                "mc_algo": (["mc", "dmc"], {"default": "mc"}),
                "num_chunks": ("INT", {"default": 8000, "min": 1000, "max": 100000}),
            },
            "optional": {
                "enable_flash_vdm": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)  # Changed from "mesh" to "trimesh" for compatibility
    FUNCTION = "decode"
    CATEGORY = "Hunyuan3D21/Decoding"

    def decode(self, vae, latents, box_v, octree_resolution, mc_level, mc_algo, num_chunks, enable_flash_vdm=True):
        torch = _lazy_import("torch")
        Trimesh = _lazy_import("trimesh")

        device = mm.get_torch_device()
        vae.to(device)

        # Enable flash VDM decoder (optimized marching cubes)
        vae.enable_flashvdm_decoder(enabled=enable_flash_vdm, mc_algo=mc_algo)

        # Decode latents through VAE
        latents = vae.decode(latents)

        # Extract mesh using marching cubes (returns list, take first element)
        outputs = vae.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=True,
        )[0]  # Extract first element from list

        # Convert to trimesh format (flip faces for correct winding order)
        outputs.mesh_f = outputs.mesh_f[:, ::-1]
        mesh_output = Trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f)

        print(f"Decoded mesh with {mesh_output.vertices.shape[0]} vertices and {mesh_output.faces.shape[0]} faces")

        gc.collect()

        return (mesh_output,)


# =============================================================================
# Node 8: PreviewTrimesh
# =============================================================================

class PreviewTrimesh:
    """
    Interactive 3D mesh preview embedded directly in ComfyUI.

    Uses PyVista to generate an interactive HTML viewer that displays
    in the ComfyUI interface with full rotate/zoom/pan controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "show_edges": ("BOOLEAN", {"default": True}),
                "color": (["default", "blue", "red", "green", "gold"], {"default": "default"}),
                "background": (["white", "black", "gradient"], {"default": "white"}),
            }
        }

    OUTPUT_NODE = True  # Makes this a preview/output node
    RETURN_TYPES = ()   # No outputs

    FUNCTION = "preview"
    CATEGORY = "Hunyuan3D21/Visualization"

    def preview(self, trimesh, show_edges=True, color="default", background="white"):
        import os
        from pathlib import Path

        try:
            import pyvista as pv
        except ImportError:
            print("⚠️  PyVista not installed. Install with: pip install pyvista")
            print("   Falling back to simple file save...")
            return self._fallback_save(trimesh)

        # Convert trimesh to pyvista
        vertices = trimesh.vertices
        faces = trimesh.faces

        # PyVista expects faces in format: [n_points, p1, p2, p3, n_points, p1, p2, p3, ...]
        pv_faces = []
        for face in faces:
            pv_faces.append(3)  # Triangle
            pv_faces.extend(face)

        mesh_pv = pv.PolyData(vertices, pv_faces)

        # Create plotter
        plotter = pv.Plotter(notebook=False, off_screen=True)

        # Set background
        if background == "white":
            plotter.set_background("white")
        elif background == "black":
            plotter.set_background("black")
        else:  # gradient
            plotter.set_background("white", top="lightblue")

        # Color mapping
        color_map = {
            "default": "#B8860B",  # GoldenRod
            "blue": "#4169E1",     # RoyalBlue
            "red": "#DC143C",      # Crimson
            "green": "#32CD32",    # LimeGreen
            "gold": "#FFD700",     # Gold
        }
        mesh_color = color_map.get(color, color_map["default"])

        # Add mesh to plotter
        plotter.add_mesh(
            mesh_pv,
            color=mesh_color,
            show_edges=show_edges,
            edge_color="black" if background == "white" else "white",
            lighting=True,
            smooth_shading=True,
        )

        # Set camera to show the whole mesh
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.2)

        # Generate interactive HTML
        html_dir = Path(folder_paths.get_temp_directory())
        html_path = html_dir / "preview_trimesh.html"

        # Export to HTML with embedded viewer
        plotter.export_html(str(html_path), backend='pythreejs')

        # Read the HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        plotter.close()

        print(f"✅ Interactive 3D preview generated")
        print(f"   Vertices: {len(trimesh.vertices)}, Faces: {len(trimesh.faces)}")
        print(f"   HTML saved: {html_path}")

        # Return HTML for embedding in ComfyUI
        return {
            "ui": {
                "html_viewer": [{
                    "html_path": str(html_path),
                    "html_content": html_content,
                    "mesh_info": {
                        "vertices": len(trimesh.vertices),
                        "faces": len(trimesh.faces)
                    }
                }]
            }
        }

    def _fallback_save(self, trimesh):
        """Fallback when PyVista is not available - just save the mesh"""
        import os
        from pathlib import Path

        temp_dir = Path(folder_paths.get_temp_directory())
        glb_path = temp_dir / "preview_trimesh.glb"

        trimesh.export(str(glb_path), file_type='glb')

        print(f"✅ Mesh saved (PyVista not available): {glb_path}")
        print(f"   Vertices: {len(trimesh.vertices)}, Faces: {len(trimesh.faces)}")

        return {
            "ui": {
                "text": [f"Mesh saved to: {glb_path}\nInstall PyVista for interactive preview: pip install pyvista"]
            }
        }


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "PrepareImageForDINO": PrepareImageForDINO,
    "LoadDinoModel": LoadDinoModel,
    "EncodeDINO": EncodeDINO,
    "LoadHunyuanDiT": LoadHunyuanDiT,
    "LoadHunyuanVAE": LoadHunyuanVAE,
    "Hy3DGenerateLatents": Hy3DGenerateLatents,
    "Hy3DDecodeLatents": Hy3DDecodeLatents,
    "Hy3DImageToLatents": Hy3DImageToLatents,
    "PreviewTrimesh": PreviewTrimesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrepareImageForDINO": "Prepare Image for DINO",
    "LoadDinoModel": "Load DINO v2 Model",
    "EncodeDINO": "Encode with DINO v2",
    "LoadHunyuanDiT": "Load Hunyuan DiT Model",
    "LoadHunyuanVAE": "Load Hunyuan VAE",
    "Hy3DGenerateLatents": "Hunyuan 3D: Generate Latents",
    "Hy3DDecodeLatents": "Hunyuan 3D: Decode Latents",
    "Hy3DImageToLatents": "Hunyuan 3D: Image to Latents (Simple)",
    "PreviewTrimesh": "Preview Trimesh",
}
