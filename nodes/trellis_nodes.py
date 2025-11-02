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
        return {
            "required": {
                "clip_model": (folder_paths.get_filename_list("text_encoders"),),
            }
        }

    RETURN_TYPES = ("CLIP_TRELLIS",)
    RETURN_NAMES = ("clip_model",)
    FUNCTION = "load_clip"
    CATEGORY = "Trellis/Loaders"

    def load_clip(self, clip_model):
        """Load CLIP model from models/text_encoders/ directory"""
        model_path = folder_paths.get_full_path("text_encoders", clip_model)

        if not os.path.exists(model_path):
            raise ValueError(f"CLIP model not found at: {model_path}")

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

    def load_trellis(self, model_type, attn_backend, spconv_algo):
        """Load Trellis model from HuggingFace repository (automatic repo selection)"""
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

        print(f"✓ Trellis {pipeline_type} model loaded successfully from {repo}")
        return ({"model": pipeline, "type": pipeline_type},)


# ============================================================================
# MONOLITHIC SAMPLER NODES - DEPRECATED (use granular pipeline nodes instead)
# ============================================================================
# These nodes are kept for backward compatibility with existing workflows.
# For new workflows, use the granular pipeline nodes in "Trellis/Pipeline" category.

class Trellis_ImageTo3D_Sampler:
    """
    DEPRECATED: Use granular pipeline nodes instead!

    This monolithic sampler combines all steps: preprocessing, conditioning,
    sparse structure sampling, SLAT sampling, decoding, and export.

    For fine-grained control, use these nodes instead:
    - Trellis_Image_Preprocessor
    - Trellis_Image_Conditioning
    - Trellis_SparseStructure_Sampler
    - Trellis_SLAT_Sampler
    - Trellis_SLAT_Decoder
    - Trellis_Export_GLB
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dinov2_model": ("DINOV2",),
                "trellis_model": ("MODEL_TRELLIS",),
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "slat_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "slat_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "preprocess_image": ("BOOLEAN", {"default": False},),
                "texture_size": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 512, "display": "number"}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 0.98, "step": 0.01}),
                "mode": (["fast", "opt"],),
                "multi_image": ("BOOLEAN", {"default": False},),
                "multiimage_algo": (["multidiffusion", "stochastic"],),
                "gaussians2ply": ("BOOLEAN", {"default": False},),
                "covert2video": ("BOOLEAN", {"default": False},),
                "glb2obj": ("BOOLEAN", {"default": False},),
                "glb2fbx": ("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "sample_image_to_3d"
    CATEGORY = "Trellis/Samplers"

    def sample_image_to_3d(self, dinov2_model, trellis_model, image, seed, cfg, steps, slat_cfg, slat_steps,
                           preprocess_image, texture_size, mesh_simplify, mode, multi_image, multiimage_algo,
                           gaussians2ply, covert2video, glb2obj, glb2fbx):
        """Sample 3D model from image using Trellis image-to-3D pipeline"""

        # Verify model type
        if trellis_model.get("type") != "image":
            raise ValueError("This sampler requires an image-to-3D Trellis model")

        # Get models
        dinov2 = dinov2_model["model"]
        dinov2_transform = dinov2_model["transform"]
        pipeline = trellis_model["model"]

        # Inject DinoV2 model into pipeline
        pipeline.models['image_cond_model'] = dinov2
        pipeline.image_cond_model_transform = dinov2_transform
        pipeline.dino_model_load = dinov2

        # Process input images
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image input must be a torch tensor")

        image_list, image_batch = tensor2imglist(image)

        if multi_image and image_batch % 3 == 0:
            print("********Inferring multi-image, like three views******")
            image_list = [image_list[i:i + 3] for i in range(0, len(image_list), 3)]
            is_multiimage = True
        else:
            is_multiimage = False

        trial_id = str(uuid.uuid4())

        # Generate 3D models
        output_path = []
        print(f"[Debug] Starting 3D generation for {len(image_list)} image(s)")

        for i, img in enumerate(image_list):
            print(f"[Debug] Processing image {i+1}/{len(image_list)}")

            print(f"[Debug] Moving pipeline to CUDA...")
            pipeline.cuda()

            print(f"[Debug] Calling image_to_3d()...")
            try:
                glb_path = image_to_3d(pipeline, img, preprocess_image, covert2video, trial_id, seed, cfg, steps,
                                       slat_cfg, slat_steps, mesh_simplify, texture_size, mode, is_multiimage,
                                       gaussians2ply, multiimage_algo)
                print(f"[Debug] ✓ image_to_3d completed: {glb_path}")
            except Exception as e:
                print(f"[Debug] ❌ image_to_3d failed with exception: {type(e).__name__}: {e}")
                raise

            output_path.append(glb_path)
            print(f"[Debug] Added to output_path, total outputs: {len(output_path)}")

        print(f"[Debug] All images processed, moving pipeline to CPU...")
        pipeline.cpu()
        print(f"[Debug] Cleaning up CUDA memory...")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Debug] Memory cleanup complete")

        # Post-processing: convert formats if requested
        output_path = self._convert_output_formats(output_path, glb2obj, glb2fbx)

        model_path = '\n'.join(output_path)
        return (model_path,)

    def _convert_output_formats(self, output_path, glb2obj, glb2fbx):
        """Convert output files to requested formats"""
        if glb2obj:
            obj_paths = []
            for path in output_path:
                obj_path = os.path.join(os.path.split(path)[0], os.path.split(path)[1].replace(".glb", ".obj"))
                glb2obj_(path, obj_path)
                obj_paths.append(obj_path)
            if glb2fbx:
                fbx_paths = []
                for i in obj_paths:
                    fbx_path = os.path.join(os.path.split(i)[0], os.path.split(i)[1].replace(".obj", ".fbx"))
                    obj2fbx_(i, fbx_path)
                    fbx_paths.append(fbx_path)
                output_path = fbx_paths
            else:
                output_path = obj_paths
        else:
            if glb2fbx:
                obj_paths = []
                fbx_paths = []
                for path in output_path:
                    obj_path = os.path.join(os.path.split(path)[0], os.path.split(path)[1].replace(".glb", ".obj"))
                    glb2obj_(path, obj_path)
                    obj_paths.append(obj_path)
                for i in obj_paths:
                    fbx_path = os.path.join(os.path.split(i)[0], os.path.split(i)[1].replace(".obj", ".fbx"))
                    obj2fbx_(i, fbx_path)
                    fbx_paths.append(fbx_path)
                output_path = fbx_paths
        return output_path


class Trellis_TextTo3D_Sampler:
    """
    DEPRECATED: Use granular pipeline nodes instead!

    This monolithic sampler combines all steps: text conditioning,
    sparse structure sampling, SLAT sampling, decoding, and export.

    For fine-grained control, use these nodes instead:
    - Trellis_Text_Conditioning
    - Trellis_SparseStructure_Sampler
    - Trellis_SLAT_Sampler
    - Trellis_SLAT_Decoder
    - Trellis_Export_GLB
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_model": ("CLIP_TRELLIS",),
                "trellis_model": ("MODEL_TRELLIS",),
                "prompt": ("STRING", {"multiline": True, "default": "Rugged, metallic texture with orange and white paint finish, suggesting a durable, industrial feel."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "slat_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "slat_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "texture_size": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 512, "display": "number"}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 0.98, "step": 0.01}),
                "mode": (["fast", "opt"],),
                "gaussians2ply": ("BOOLEAN", {"default": False},),
                "covert2video": ("BOOLEAN", {"default": False},),
                "glb2obj": ("BOOLEAN", {"default": False},),
                "glb2fbx": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "mesh_path": ("STRING", {"default": ""}),  # For variant mode
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "sample_text_to_3d"
    CATEGORY = "Trellis/Samplers"

    def sample_text_to_3d(self, clip_model, trellis_model, prompt, seed, cfg, steps, slat_cfg, slat_steps,
                          texture_size, mesh_simplify, mode, gaussians2ply, covert2video, glb2obj, glb2fbx,
                          mesh_path=""):
        """Sample 3D model from text using Trellis text-to-3D pipeline"""

        # Verify model type
        if trellis_model.get("type") != "text":
            raise ValueError("This sampler requires a text-to-3D Trellis model")

        # Get models
        clip = clip_model["model"]
        tokenizer = clip_model["tokenizer"]
        pipeline = trellis_model["model"]

        # Inject CLIP model into pipeline
        pipeline.clip_model = clip.cuda()
        pipeline.text_cond_model = {
            'model': clip.cuda(),
            'tokenizer': tokenizer,
        }

        # Encode empty string for negative conditioning
        import torch.nn.functional as F
        tokens = tokenizer([''], max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        pipeline.text_cond_model['null_cond'] = clip(input_ids=tokens['input_ids'].cuda()).last_hidden_state

        trial_id = str(uuid.uuid4())

        # Generate 3D model
        pipeline.cuda()
        if mesh_path and mesh_path.strip():
            if not mesh_path.endswith(".ply"):
                raise ValueError("mesh_path must be .ply format")
            import open3d as o3d

            base_mesh = o3d.io.read_triangle_mesh(mesh_path)
            outputs = pipeline.run_variant(
                base_mesh,
                prompt=prompt,
                seed=seed,
                slat_sampler_params={
                    "steps": slat_steps,
                    "cfg_strength": slat_cfg,
                },
            )
        else:
            outputs = pipeline.run(
                prompt,
                seed=seed,
                formats=["gaussian", "mesh"],
                sparse_structure_sampler_params={
                    "steps": steps,
                    "cfg_strength": cfg,
                },
                slat_sampler_params={
                    "steps": slat_steps,
                    "cfg_strength": slat_cfg,
                },
            )

        output_path = prepare_output(outputs, covert2video, trial_id, False, gaussians2ply, mesh_simplify,
                                      texture_size, mode)
        output_path = [output_path]

        pipeline.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Post-processing: convert formats if requested
        output_path = self._convert_output_formats(output_path, glb2obj, glb2fbx)

        model_path = '\n'.join(output_path)
        return (model_path,)

    def _convert_output_formats(self, output_path, glb2obj, glb2fbx):
        """Convert output files to requested formats"""
        if glb2obj:
            obj_paths = []
            for path in output_path:
                obj_path = os.path.join(os.path.split(path)[0], os.path.split(path)[1].replace(".glb", ".obj"))
                glb2obj_(path, obj_path)
                obj_paths.append(obj_path)
            if glb2fbx:
                fbx_paths = []
                for i in obj_paths:
                    fbx_path = os.path.join(os.path.split(i)[0], os.path.split(i)[1].replace(".obj", ".fbx"))
                    obj2fbx_(i, fbx_path)
                    fbx_paths.append(fbx_path)
                output_path = fbx_paths
            else:
                output_path = obj_paths
        else:
            if glb2fbx:
                obj_paths = []
                fbx_paths = []
                for path in output_path:
                    obj_path = os.path.join(os.path.split(path)[0], os.path.split(path)[1].replace(".glb", ".obj"))
                    glb2obj_(path, obj_path)
                    obj_paths.append(obj_path)
                for i in obj_paths:
                    fbx_path = os.path.join(os.path.split(i)[0], os.path.split(i)[1].replace(".obj", ".fbx"))
                    obj2fbx_(i, fbx_path)
                    fbx_paths.append(fbx_path)
                output_path = fbx_paths
        return output_path


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

    # Monolithic samplers (for backward compatibility)
    "Trellis_ImageTo3D_Sampler": Trellis_ImageTo3D_Sampler,
    "Trellis_TextTo3D_Sampler": Trellis_TextTo3D_Sampler,

    # Granular pipeline nodes (new)
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

    # Monolithic samplers
    "Trellis_ImageTo3D_Sampler": "Trellis Image-to-3D Sampler",
    "Trellis_TextTo3D_Sampler": "Trellis Text-to-3D Sampler",

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

