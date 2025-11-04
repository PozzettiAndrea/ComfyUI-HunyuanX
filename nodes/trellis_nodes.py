# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import gc
import os
import torch
import uuid
import random
from transformers import CLIPTextModel, AutoTokenizer, CLIPTextConfig
from contextlib import nullcontext
from safetensors.torch import load_file
from torchvision import transforms
from PIL import Image

from .nodeutils.trellis_utils import image_to_3d,prepare_output
from .lib.trellis.pipelines import TrellisImageTo3DPipeline,TrellisTextTo3DPipeline
from .nodeutils.trellis_utils import glb2obj_,obj2fbx_,tensor2imglist,pre_img
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# MODEL LOADER NODES - Transparent model loading with explicit inputs
# ============================================================================

class Load_DinoV2_Model:
    """
    Loads DinoV2 model with registers for image conditioning in Trellis image-to-3D pipeline.

    Uses torch.hub to download the correct register variants (dinov2_vit*14_reg) that TRELLIS was trained with.
    These models have additional register tokens that improve feature quality significantly.
    """

    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"], {
                    "default": "dinov2_vitl14_reg",
                    "tooltip": "DinoV2 model with registers (TRELLIS uses vitl14_reg by default)"
                }),
                "keep_in_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in memory between runs (faster)"
                }),
            }
        }

    RETURN_TYPES = ("DINOV2",)
    RETURN_NAMES = ("dinov2_model",)
    FUNCTION = "load_dinov2"
    CATEGORY = "Trellis/Loaders"

    def load_dinov2(self, model_variant, keep_in_memory=True):
        """Load DinoV2 register model from torch.hub - auto-downloads"""
        import torch

        # Clear cache if keep_in_memory is False
        if not keep_in_memory and self.model is not None:
            print(f"🗑️ Clearing DinoV2 model from cache")
            del self.model
            self.model = None
            self.current_model_name = None

        # Check if already loaded
        if self.model is not None and self.current_model_name == model_variant:
            print(f"⚡ Using cached DinoV2 model ({model_variant})")
        else:
            print(f"📥 Downloading DinoV2 from Facebook Research: {model_variant}")
            print(f"   Source: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/")
            try:
                # Download DinoV2 with registers from torch.hub
                # This downloads from https://dl.fbaipublicfiles.com/dinov2/
                dinov2_model = torch.hub.load('facebookresearch/dinov2', model_variant, pretrained=True)
                dinov2_model.eval()

                self.model = dinov2_model
                self.current_model_name = model_variant

                # Print model info
                total_params = sum(p.numel() for p in dinov2_model.parameters())
                print(f"✅ Loaded DinoV2 with registers: {model_variant}")
                print(f"   Parameters: {total_params:,}")
                print(f"   Register tokens: 4 (improves feature quality)")
            except Exception as e:
                raise ValueError(f"Failed to load DinoV2 from torch.hub: {e}\n"
                                f"Make sure you have internet connection for first-time download.")

        # Prepare transform (TRELLIS expects normalized images)
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return ({"model": self.model, "transform": transform},)


class Load_CLIP_Trellis:
    """Loads CLIP model for text conditioning in Trellis text-to-3D pipeline"""

    @classmethod
    def INPUT_TYPES(cls):
        available_models = folder_paths.get_filename_list("text_encoders")

        # If no models found, provide auto-download option
        if not available_models:
            available_models = ["clip_l.safetensors"]

        return {
            "required": {
                "clip_model": (available_models, {
                    "default": "clip_l.safetensors" if "clip_l.safetensors" in available_models else available_models[0]
                }),
            }
        }

    RETURN_TYPES = ("CLIP_TRELLIS",)
    RETURN_NAMES = ("clip_model",)
    FUNCTION = "load_clip"
    CATEGORY = "Trellis/Loaders"

    def _download_clip_model(self, clip_model):
        """Download CLIP model from HuggingFace if not found locally"""
        try:
            from huggingface_hub import hf_hub_download
            import shutil
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed. Please install it:\n"
                "pip install huggingface_hub"
            )

        # Use ComfyUI's official CLIP-L model
        repo_id = "comfyanonymous/flux_text_encoders"
        filename = "clip_l.safetensors"

        print(f"📥 CLIP model not found locally, downloading from HuggingFace...")
        print(f"   Repository: {repo_id}")
        print(f"   File: {filename} (~246 MB)")
        print(f"   This is a one-time download, please wait...")

        try:
            # Download to HuggingFace cache
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model"
            )

            # Copy to ComfyUI models/clip folder
            clip_dir = os.path.join(folder_paths.models_dir, "clip")
            os.makedirs(clip_dir, exist_ok=True)
            local_path = os.path.join(clip_dir, filename)

            print(f"📦 Copying to ComfyUI models directory...")
            shutil.copy(downloaded_path, local_path)

            print(f"✅ CLIP model downloaded successfully!")
            print(f"   Saved to: {local_path}")

            return local_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download CLIP model from HuggingFace: {e}\n\n"
                f"Manual download instructions:\n"
                f"1. Download: https://huggingface.co/{repo_id}/resolve/main/{filename}\n"
                f"2. Place in: {os.path.join(folder_paths.models_dir, 'clip')}\n"
                f"3. Restart ComfyUI"
            )

    def load_clip(self, clip_model):
        """Load CLIP model from models/text_encoders/ directory"""
        model_path = folder_paths.get_full_path("text_encoders", clip_model)

        # Auto-download if missing (get_full_path returns None when not found)
        if model_path is None or not os.path.exists(model_path):
            # Construct expected path for error message
            search_dirs = folder_paths.get_folder_paths("text_encoders")
            expected_path = os.path.join(search_dirs[0], clip_model) if search_dirs else "models/clip/"
            print(f"⚠️  CLIP model not found at: {expected_path}")
            model_path = self._download_clip_model(clip_model)

        print(f"Loading CLIP model from: {model_path}")

        # Load using the same method as trellis_text_to_3d.py
        # Note: This requires the clip/ config directory for tokenizer
        meshcraft_root = os.path.dirname(current_path)  # Go up from nodes/ to MeshCraft root
        config_path = os.path.join(meshcraft_root, "nodes", "lib", "clip")

        if not os.path.exists(config_path):
            raise ValueError(f"CLIP config directory not found at: {config_path}\nThis is needed for the tokenizer.")

        # Load model
        is_accelerate_available = False
        try:
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
            is_accelerate_available = True
        except:
            pass

        ctx = init_empty_weights if is_accelerate_available else nullcontext
        with ctx():
            text_encoder_config = CLIPTextConfig.from_pretrained(config_path, local_files_only=True)

        model = CLIPTextModel(text_encoder_config)
        tokenizer = AutoTokenizer.from_pretrained(config_path)

        clip_dict = load_file(model_path)
        model.load_state_dict(clip_dict, strict=False)
        del clip_dict
        torch.cuda.empty_cache()
        model.eval()

        print(f"✓ CLIP model loaded successfully")
        return ({"model": model, "tokenizer": tokenizer},)


class Load_Trellis_Model:
    """Loads Trellis 3D generation model from HuggingFace (Microsoft repos)"""

    # Hardcoded Microsoft repository paths
    REPO_MAP = {
        "image-to-3d": "microsoft/TRELLIS-image-large",
        "text-to-3d": "microsoft/TRELLIS-text-large"
    }

    def __init__(self):
        self.cached_pipeline = None
        self.cached_config = None  # Store (model_type, attn_backend, spconv_algo)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["image-to-3d", "text-to-3d"],),
                "attn_backend": (["flash-attn", "sdpa", "naive", "xformers"],),
                "spconv_algo": (["auto", "flash-native"],),
            }
        }

    RETURN_TYPES = ("MODEL_TRELLIS",)
    RETURN_NAMES = ("trellis_model",)
    FUNCTION = "load_trellis"
    CATEGORY = "Trellis/Loaders"

    @classmethod
    def IS_CHANGED(cls, model_type, attn_backend, spconv_algo):
        """Force reload when attention config changes"""
        return f"{model_type}_{attn_backend}_{spconv_algo}"

    def load_trellis(self, model_type, attn_backend, spconv_algo):
        """Load Trellis model from HuggingFace repository (automatic repo selection)"""

        # Check if we can use cached pipeline
        current_config = (model_type, attn_backend, spconv_algo)
        if self.cached_pipeline is not None and self.cached_config == current_config:
            pipeline_type = "text" if model_type == "text-to-3d" else "image"
            print(f"⚡ Using cached Trellis {pipeline_type} pipeline ({attn_backend}, {spconv_algo})")
            return ({"model": self.cached_pipeline, "type": pipeline_type},)

        # Clear old cache if config changed
        if self.cached_pipeline is not None:
            print(f"🗑️ Clearing old Trellis pipeline from cache (config changed)")
            del self.cached_pipeline
            self.cached_pipeline = None
            torch.cuda.empty_cache()
            gc.collect()

        # Set environment variables for backend configuration
        # Options: flash-attn (uses flash-attention directly), sdpa (PyTorch native),
        #          naive (pure PyTorch), xformers (requires flash-attn <=2.8.2)
        os.environ['ATTN_BACKEND'] = attn_backend.replace('-', '_')  # flash-attn -> flash_attn

        if spconv_algo == "auto":
            os.environ['SPCONV_ALGO'] = 'auto'
        else:
            os.environ['SPCONV_ALGO'] = 'native'

        print(f"[Trellis] Attention backend: {attn_backend}, Sparse convolution: {spconv_algo}")

        # Automatically select the correct Microsoft repository
        repo = self.REPO_MAP[model_type]
        pipeline_type = "text" if model_type == "text-to-3d" else "image"

        print(f"Loading Trellis {pipeline_type} model from: {repo}")

        # Load the appropriate pipeline WITHOUT conditioning models
        # We'll pass those separately later via the sampler nodes
        if pipeline_type == "image":
            # Load base pipeline without initializing image_cond_model
            pipeline = TrellisImageTo3DPipeline.from_pretrained(repo, skip_cond_model=True)
        else:
            # Load base pipeline without initializing text_cond_model
            pipeline = TrellisTextTo3DPipeline.from_pretrained(repo, skip_cond_model=True)

        # Cache the loaded pipeline
        self.cached_pipeline = pipeline
        self.cached_config = current_config

        print(f"✓ Trellis {pipeline_type} model loaded and cached")
        return ({"model": pipeline, "type": pipeline_type},)



class Trellis_multiimage_loader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_a": ("IMAGE",),
                             },
                "optional": {"image_b": ("IMAGE",),
                             "image_c": ("IMAGE",),}
                }

    RETURN_TYPES = ("IMAGE",)
    ETURN_NAMES = ("image",)
    FUNCTION = "main_batch"
    CATEGORY = "Trellis"

    def main_batch(self, image_a, **kwargs):
        image_b = kwargs.get("image_b")
        image_c = kwargs.get("image_c")
        _,height_a,_,_ = image_a.shape
        if isinstance(image_b, torch.Tensor) and isinstance(image_c, torch.Tensor):
            _, height_b, _, _ = image_b.shape
            _, height_c, _, _ = image_c.shape
            height = max(height_a, height_b, height_c)
            img_list=[pre_img(image_a, height),pre_img(image_b, height),pre_img(image_c, height)]
            image = torch.cat(img_list, dim=0)
        elif isinstance(image_b, torch.Tensor) and not isinstance(image_c, torch.Tensor):
            _, height_b, _, _ = image_b.shape
            height = max(height_a, height_b,)
            img_list = [pre_img(image_a, height), pre_img(image_b, height)]
            image = torch.cat(img_list, dim=0)
        elif not isinstance(image_b, torch.Tensor) and isinstance(image_c, torch.Tensor):
            _, height_c, _, _ = image_c.shape
            height = max(height_a, height_c, )
            img_list = [pre_img(image_a, height), pre_img(image_b, height)]
            image = torch.cat(img_list, dim=0)
        else:
            image=image_a

        return (image,)


# ============================================================================
# GRANULAR PIPELINE NODES - Ultra-fine control over each pipeline stage
# ============================================================================

class Trellis_Image_Preprocessor:
    """
    Preprocesses images for Trellis: removes background, crops, resizes.
    Optional stage - can be bypassed by passing raw images to conditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_model": ("MODEL_TRELLIS",),
                "image": ("IMAGE",),
                "enable_preprocessing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed_image",)
    FUNCTION = "preprocess"
    CATEGORY = "Trellis/Pipeline"

    def preprocess(self, trellis_model, image, enable_preprocessing):
        """Preprocess image: background removal, cropping, resizing to 518x518"""
        pipeline = trellis_model["model"]

        if not enable_preprocessing:
            # Just pass through
            return (image,)

        # Convert tensor to PIL images
        image_list, _ = tensor2imglist(image)

        # Preprocess each image
        preprocessed_images = []
        for img_pil in image_list:
            preprocessed_img = pipeline.preprocess_image(img_pil)
            # Convert back to tensor
            img_np = np.array(preprocessed_img).astype(np.float32) / 255.0
            if img_np.shape[2] == 4:  # RGBA
                img_np = img_np[:, :, :3]  # Drop alpha
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            preprocessed_images.append(img_tensor)

        output_tensor = torch.cat(preprocessed_images, dim=0)
        return (output_tensor,)


class Trellis_Image_Conditioning:
    """
    Encodes images using DinoV2 to create conditioning for 3D generation.
    Output: CONDITIONING dict with 'cond' and 'neg_cond' tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dinov2_model": ("DINOV2",),
                "trellis_model": ("MODEL_TRELLIS",),
                "image": ("IMAGE",),  # Can be preprocessed or raw
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Trellis/Pipeline"

    def encode(self, dinov2_model, trellis_model, image):
        """Encode image to conditioning using DinoV2"""
        dinov2 = dinov2_model["model"]
        dinov2_transform = dinov2_model["transform"]
        pipeline = trellis_model["model"]

        # Inject DinoV2 model into pipeline temporarily
        pipeline.models['image_cond_model'] = dinov2
        pipeline.image_cond_model_transform = dinov2_transform
        pipeline.dino_model_load = dinov2
        pipeline.cuda()

        # Convert tensor to PIL images
        image_list, _ = tensor2imglist(image)

        # Get conditioning
        cond_dict = pipeline.get_cond(image_list)

        pipeline.cpu()
        torch.cuda.empty_cache()

        return (cond_dict,)


class Trellis_Text_Conditioning:
    """
    Encodes text using CLIP to create conditioning for 3D generation.
    Output: CONDITIONING dict with 'cond' and 'neg_cond' tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_model": ("CLIP_TRELLIS",),
                "trellis_model": ("MODEL_TRELLIS",),
                "prompt": ("STRING", {"multiline": True, "default": "a 3D model"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Trellis/Pipeline"

    def encode(self, clip_model, trellis_model, prompt):
        """Encode text to conditioning using CLIP"""
        clip = clip_model["model"]
        tokenizer = clip_model["tokenizer"]
        pipeline = trellis_model["model"]

        # Move CLIP to GPU
        clip = clip.cuda()

        # Encode negative conditioning (empty string)
        neg_tokens = tokenizer([''], max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        neg_cond = clip(input_ids=neg_tokens['input_ids'].cuda()).last_hidden_state

        # Encode positive conditioning (prompt)
        pos_tokens = tokenizer([prompt], max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        pos_cond = clip(input_ids=pos_tokens['input_ids'].cuda()).last_hidden_state

        # Create conditioning dict matching the expected format
        cond_dict = {
            'cond': pos_cond,
            'neg_cond': neg_cond,
        }

        # Cleanup
        clip.cpu()
        torch.cuda.empty_cache()

        return (cond_dict,)


"""
# DISABLED: Combine_Conditioning is fundamentally flawed
# DinoV2 (image) and CLIP (text) embeddings come from different feature spaces
# and cannot be meaningfully combined through averaging or concatenation.
#
# This would require a model trained on BOTH simultaneously, which doesn't exist.
# Keep image-to-3d and text-to-3d pipelines separate.

class Trellis_Combine_Conditioning:
    # [CODE COMMENTED OUT - See explanation above]
    pass
"""


class Trellis_SparseStructure_Sampler:
    """
    Samples sparse 3D structure coordinates from conditioning.
    This is the first generation stage - creates a coarse voxel occupancy grid.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_model": ("MODEL_TRELLIS",),
                "conditioning": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "randomize_seed": ("BOOLEAN", {"default": False}),
                "guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("COORDS",)
    RETURN_NAMES = ("sparse_coords",)
    FUNCTION = "sample"
    CATEGORY = "Trellis/Pipeline"

    def sample(self, trellis_model, conditioning, seed, randomize_seed, guidance_strength, sampling_steps, num_samples):
        """Sample sparse structure coordinates"""
        pipeline = trellis_model["model"]
        pipeline.cuda()

        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)

        torch.manual_seed(seed)

        sampler_params = {
            "steps": sampling_steps,
            "cfg_strength": guidance_strength,
        }

        coords = pipeline.sample_sparse_structure(conditioning, num_samples, sampler_params)

        pipeline.cpu()
        torch.cuda.empty_cache()

        return (coords,)


class Trellis_SLAT_Sampler:
    """
    Samples Structured Latent (SLAT) from sparse coordinates and conditioning.
    This is the second generation stage - adds fine details to the coarse structure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_model": ("MODEL_TRELLIS",),
                "conditioning": ("CONDITIONING",),
                "sparse_coords": ("COORDS",),
                "guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "cfg_interval_start": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg_interval_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_t": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("SLAT",)
    RETURN_NAMES = ("slat",)
    FUNCTION = "sample"
    CATEGORY = "Trellis/Pipeline"

    def sample(self, trellis_model, conditioning, sparse_coords, guidance_strength, sampling_steps,
               cfg_interval_start, cfg_interval_end, rescale_t):
        """Sample structured latent (SLAT) from sparse coordinates"""
        pipeline = trellis_model["model"]
        pipeline.cuda()

        sampler_params = {
            "steps": sampling_steps,
            "cfg_strength": guidance_strength,
            "cfg_interval": [cfg_interval_start, cfg_interval_end],
            "rescale_t": rescale_t,
        }

        slat = pipeline.sample_slat(conditioning, sparse_coords, sampler_params)

        pipeline.cpu()
        torch.cuda.empty_cache()

        return (slat,)


class Trellis_SLAT_Decoder:
    """
    Decodes SLAT to 3D representations: Gaussian Splatting, Mesh, and/or Radiance Field.
    You can choose which formats to decode (uncheck to skip expensive decoding).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_model": ("MODEL_TRELLIS",),
                "slat": ("SLAT",),
                "decode_gaussian": ("BOOLEAN", {"default": True}),
                "decode_mesh": ("BOOLEAN", {"default": True}),
                "decode_radiance_field": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("GAUSSIAN", "MESH", "RADIANCE_FIELD")
    RETURN_NAMES = ("gaussian", "mesh", "radiance_field")
    FUNCTION = "decode"
    CATEGORY = "Trellis/Pipeline"

    def decode(self, trellis_model, slat, decode_gaussian, decode_mesh, decode_radiance_field):
        """Decode SLAT to 3D representations"""
        pipeline = trellis_model["model"]
        pipeline.cuda()

        formats = []
        if decode_gaussian:
            formats.append("gaussian")
        if decode_mesh:
            formats.append("mesh")
        if decode_radiance_field:
            formats.append("radiance_field")

        if not formats:
            raise ValueError("Must decode at least one format (gaussian, mesh, or radiance_field)")

        outputs = pipeline.decode_slat(slat, formats)

        pipeline.cpu()
        torch.cuda.empty_cache()

        # Return None for formats that weren't decoded
        gaussian = outputs.get("gaussian", [None])[0] if "gaussian" in outputs else None
        mesh = outputs.get("mesh", [None])[0] if "mesh" in outputs else None
        radiance_field = outputs.get("radiance_field", [None])[0] if "radiance_field" in outputs else None

        return (gaussian, mesh, radiance_field)


class Trellis_Export_GLB:
    """
    Exports Gaussian + Mesh to GLB file format.
    Handles mesh simplification, texture baking, and file export.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gaussian": ("GAUSSIAN",),
                "mesh": ("MESH",),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 0.99, "step": 0.01}),
                "texture_size": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 512}),
                "texture_mode": (["fast", "opt"],),
                "filename": ("STRING", {"default": "output"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "export"
    CATEGORY = "Trellis/Export"

    def export(self, gaussian, mesh, mesh_simplify, texture_size, texture_mode, filename):
        """Export gaussian and mesh to GLB file"""
        from .lib.trellis.utils import postprocessing_utils

        trial_id = filename if filename else str(uuid.uuid4())

        # Convert to GLB
        glb = postprocessing_utils.to_glb(
            gaussian,
            mesh,
            simplify=mesh_simplify,
            texture_size=texture_size,
            mode=texture_mode,
            uv_map=f"{folder_paths.get_output_directory()}/{trial_id}_uv_map.png",
        )

        # Export to file
        prefix = ''.join(random.choice("0123456789") for _ in range(5))
        glb_path = f"{folder_paths.get_output_directory()}/{trial_id}_{prefix}.glb"
        glb.export(glb_path)

        print(f"✓ GLB exported to: {glb_path}")
        return (glb_path,)


class Trellis_Export_PLY:
    """
    Exports Gaussian Splatting to PLY file format.
    PLY format is useful for viewing in 3D viewers that support gaussian splatting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gaussian": ("GAUSSIAN",),
                "filename": ("STRING", {"default": "gaussians"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "export"
    CATEGORY = "Trellis/Export"

    def export(self, gaussian, filename):
        """Export gaussian splatting to PLY file"""
        trial_id = filename if filename else str(uuid.uuid4())
        ply_path = f"{folder_paths.get_output_directory()}/{trial_id}.ply"

        gaussian.save_ply(ply_path)

        print(f"✓ PLY exported to: {ply_path}")
        return (ply_path,)


class Trellis_Render_Video:
    """
    Renders a rotating video from Gaussian or Mesh representation.
    Useful for previewing the 3D model as a turntable video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "representation": ("GAUSSIAN,MESH",),  # Either gaussian or mesh
                "render_type": (["color", "normal"],),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "filename": ("STRING", {"default": "render"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "render"
    CATEGORY = "Trellis/Export"

    def render(self, representation, render_type, fps, filename):
        """Render video from 3D representation"""
        import imageio
        from .lib.trellis.utils import render_utils

        trial_id = filename if filename else str(uuid.uuid4())
        video_path = f"{folder_paths.get_output_directory()}/{trial_id}.mp4"

        # Render frames
        video = render_utils.render_video(representation)[render_type]

        # Save video
        imageio.mimsave(video_path, video, fps=fps)

        print(f"✓ Video rendered to: {video_path}")
        return (video_path,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # Loaders
    "Load_DinoV2_Model": Load_DinoV2_Model,
    "Load_CLIP_Trellis": Load_CLIP_Trellis,
    "Load_Trellis_Model": Load_Trellis_Model,

    # Granular pipeline nodes
    "Trellis_Image_Preprocessor": Trellis_Image_Preprocessor,
    "Trellis_Image_Conditioning": Trellis_Image_Conditioning,
    "Trellis_Text_Conditioning": Trellis_Text_Conditioning,
    # "Trellis_Combine_Conditioning": Trellis_Combine_Conditioning,  # REMOVED: Incompatible feature spaces
    "Trellis_SparseStructure_Sampler": Trellis_SparseStructure_Sampler,
    "Trellis_SLAT_Sampler": Trellis_SLAT_Sampler,
    "Trellis_SLAT_Decoder": Trellis_SLAT_Decoder,
    "Trellis_Export_GLB": Trellis_Export_GLB,
    "Trellis_Export_PLY": Trellis_Export_PLY,
    "Trellis_Render_Video": Trellis_Render_Video,

    # Utility nodes
    "Trellis_multiimage_loader": Trellis_multiimage_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "Load_DinoV2_Model": "Load DinoV2 Model (Trellis)",
    "Load_CLIP_Trellis": "Load CLIP Model (Trellis)",
    "Load_Trellis_Model": "Load Trellis Model",

    # Granular pipeline nodes
    "Trellis_Image_Preprocessor": "Image Preprocessor (Trellis)",
    "Trellis_Image_Conditioning": "Image Conditioning (Trellis)",
    "Trellis_Text_Conditioning": "Text Conditioning (Trellis)",
    # "Trellis_Combine_Conditioning": "Combine Conditioning (Trellis)",  # REMOVED
    "Trellis_SparseStructure_Sampler": "Sparse Structure Sampler (Trellis)",
    "Trellis_SLAT_Sampler": "SLAT Sampler (Trellis)",
    "Trellis_SLAT_Decoder": "SLAT Decoder (Trellis)",
    "Trellis_Export_GLB": "Export GLB (Trellis)",
    "Trellis_Export_PLY": "Export PLY (Trellis)",
    "Trellis_Render_Video": "Render Video (Trellis)",

    # Utility
    "Trellis_multiimage_loader": "Trellis Multi-Image Loader",
}

