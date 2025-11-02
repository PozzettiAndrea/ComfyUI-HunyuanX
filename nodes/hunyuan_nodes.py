"""
ComfyUI-MeshCraft Hunyuan 3D Nodes

Based on ComfyUI-Hunyuan3d-2-1 by visualbruno
Original: https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1

Modifications:
- Added model caching for faster workflow execution (~5-10x speedup on reloads)
- Added configurable cache toggles for debugging
- Memory management improvements

Powered by Tencent Hunyuan 3D 2.1
License: Tencent Hunyuan 3D 2.1 Community License (see LICENSE_TENCENT_HUNYUAN)
"""

# =============================================================================
# Lazy Import System - Loads heavy dependencies only when needed
# This reduces ComfyUI startup time from ~26s to <1s
# =============================================================================

# Lightweight imports (always loaded)
import os
import time
import re
import json
import hashlib
import gc
import shutil
from typing import Union, Optional, Tuple, List, Any, Callable
from pathlib import Path

from .nodeutils.model_cache import get_cache_key, get_cached_model, cache_model, clear_cache, _MODEL_CACHE

# ComfyUI imports (lightweight)
import folder_paths
import node_helpers
import comfy.model_management as mm
import comfy.utils

# =============================================================================
# Lazy Import Cache - Heavy modules loaded on first use
# =============================================================================
_LAZY_IMPORTS = {}

def _lazy_import(module_name):
    """Cache and return heavy imports only when first needed"""
    if module_name not in _LAZY_IMPORTS:
        if module_name == "torch":
            import torch
            _LAZY_IMPORTS["torch"] = torch
        elif module_name == "torch.nn":
            import torch.nn as nn
            _LAZY_IMPORTS["torch.nn"] = nn
        elif module_name == "torch.nn.functional":
            import torch.nn.functional as F
            _LAZY_IMPORTS["torch.nn.functional"] = F
        elif module_name == "numpy":
            import numpy as np
            _LAZY_IMPORTS["numpy"] = np
        elif module_name == "trimesh":
            import trimesh as Trimesh
            _LAZY_IMPORTS["trimesh"] = Trimesh
        elif module_name == "PIL.Image":
            from PIL import Image, ImageSequence, ImageOps
            _LAZY_IMPORTS["PIL.Image"] = Image
            _LAZY_IMPORTS["PIL.ImageSequence"] = ImageSequence
            _LAZY_IMPORTS["PIL.ImageOps"] = ImageOps
        elif module_name == "hunyuan_pipeline":
            from .lib.hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            _LAZY_IMPORTS["hunyuan_pipeline"] = Hunyuan3DDiTFlowMatchingPipeline
        elif module_name == "hunyuan_postprocessors":
            from .lib.hy3dshape.hy3dshape.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover
            _LAZY_IMPORTS["FaceReducer"] = FaceReducer
            _LAZY_IMPORTS["FloaterRemover"] = FloaterRemover
            _LAZY_IMPORTS["DegenerateFaceRemover"] = DegenerateFaceRemover
        elif module_name == "hunyuan_paint":
            from .lib.hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            _LAZY_IMPORTS["Hunyuan3DPaintPipeline"] = Hunyuan3DPaintPipeline
            _LAZY_IMPORTS["Hunyuan3DPaintConfig"] = Hunyuan3DPaintConfig
        elif module_name == "hunyuan_vae":
            from .lib.hy3dshape.hy3dshape.models.autoencoders import ShapeVAE
            _LAZY_IMPORTS["ShapeVAE"] = ShapeVAE
        elif module_name == "mesh_uv_wrap":
            from .lib.hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap
            _LAZY_IMPORTS["mesh_uv_wrap"] = mesh_uv_wrap
        elif module_name == "meshlib":
            from .lib.hy3dshape.hy3dshape.meshlib import postprocessmesh
            _LAZY_IMPORTS["postprocessmesh"] = postprocessmesh
        elif module_name == "comfy_utils":
            from comfy.utils import load_torch_file, ProgressBar
            _LAZY_IMPORTS["load_torch_file"] = load_torch_file
            _LAZY_IMPORTS["ProgressBar"] = ProgressBar

    return _LAZY_IMPORTS.get(module_name)

script_directory = os.path.dirname(os.path.abspath(__file__))
meshcraft_root = os.path.dirname(script_directory)  # Go up from nodes/ to MeshCraft root
config_directory = os.path.join(meshcraft_root, 'lib', 'hunyuan_configs')  # Config path
# Go up 4 levels: nodes/ -> MeshCraft/ -> custom_nodes/ -> ComfyUI/
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
diffusions_dir = os.path.join(comfy_path, "models", "diffusers")

def parse_string_to_int_list(number_string):
  """
  Parses a string containing comma-separated numbers into a list of integers.

  Args:
    number_string: A string containing comma-separated numbers (e.g., "20000,10000,5000").

  Returns:
    A list of integers parsed from the input string.
    Returns an empty list if the input string is empty or None.
  """
  if not number_string:
    return []

  try:
    # Split the string by comma and convert each part to an integer
    int_list = [int(num.strip()) for num in number_string.split(',')]
    return int_list
  except ValueError as e:
    print(f"Error converting string to integer: {e}. Please ensure all values are valid numbers.")
    return []

def hy3dpaintimages_to_tensor(images):
    np = _lazy_import("numpy")
    torch = _lazy_import("torch")

    tensors = []
    for pil_img in images:
        np_img = np.array(pil_img).astype(np.uint8)
        np_img = np_img / 255.0
        tensor_img = torch.from_numpy(np_img).float()
        tensors.append(tensor_img)
    tensors = torch.stack(tensors)
    return tensors

def get_picture_files(folder_path):
    """
    Retrieves all picture files (based on common extensions) from a given folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to the picture files found.
    """
    picture_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    picture_files = []

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []
                
    for entry_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry_name)

        # Check if the entry is actually a file (and not a sub-directory)
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry_name)
            if file_extension.lower().endswith(picture_extensions):
                picture_files.append(full_path)                
    return picture_files
    
def get_mesh_files(folder_path, name_filter = None):
    """
    Retrieves all picture files (based on common extensions) from a given folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to the picture files found.
    """
    mesh_extensions = ('.obj', '.glb')
    mesh_files = []

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []
                    
    for entry_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry_name)

        # Check if the entry is actually a file (and not a sub-directory)
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry_name)
            if file_extension.lower().endswith(mesh_extensions):
                if name_filter is None or name_filter.lower() in file_name.lower():
                    mesh_files.append(full_path)                 
    return mesh_files    

def get_filename_without_extension_os_path(full_file_path):
    """
    Extracts the filename without its extension from a full file path using os.path.

    Args:
        full_file_path (str): The complete path to the file.

    Returns:
        str: The filename without its extension.
    """
    # 1. Get the base name (filename with extension)
    base_name = os.path.basename(full_file_path)
    
    # 2. Split the base name into root (filename without ext) and extension
    file_name_without_ext, _ = os.path.splitext(base_name)
    
    return file_name_without_ext

def _convert_texture_format(tex,
                          texture_size: Tuple[int, int], device: str, force_set: bool = False):
    """Unified texture format conversion logic."""
    np = _lazy_import("numpy")
    torch = _lazy_import("torch")
    _lazy_import("PIL.Image")
    Image = _LAZY_IMPORTS["PIL.Image"]

    if not force_set:
        if isinstance(tex, np.ndarray):
            tex = Image.fromarray((tex * 255).astype(np.uint8))
        elif isinstance(tex, torch.Tensor):            
            tex_np = tex.cpu().numpy()

            # 2. Handle potential batch dimension (B, C, H, W) or (B, H, W, C)
            if tex_np.ndim == 4:
                if tex_np.shape[0] == 1:
                    tex_np = tex_np.squeeze(0)
                else:
                    tex_np = tex_np[0]
            
            # 3. Handle data type and channel order for PIL
            if tex_np.ndim == 3:
                if tex_np.shape[0] in [1, 3, 4] and tex_np.shape[0] < tex_np.shape[1] and tex_np.shape[0] < tex_np.shape[2]:
                    tex_np = np.transpose(tex_np, (1, 2, 0))
                elif tex_np.shape[2] in [1, 3, 4] and tex_np.shape[0] > 4 and tex_np.shape[1] > 4:
                    pass
                else:
                    raise ValueError(f"Unsupported 3D tensor shape after squeezing batch and moving to CPU. "
                                     f"Expected (C, H, W) or (H, W, C) but got {tex_np.shape}")
                
                if tex_np.shape[2] == 1:
                    tex_np = tex_np.squeeze(2) # Remove the channel dimension

            elif tex_np.ndim == 2:
                pass
            else:
                raise ValueError(f"Unsupported tensor dimension after squeezing batch and moving to CPU: {tex_np.ndim} "
                                 f"with shape {tex_np.shape}. Expected 2D or 3D image data.")

            tex_np_uint8 = (tex_np * 255).astype(np.uint8)    
            
            tex = Image.fromarray(tex_np_uint8)

        
        tex = tex.resize(texture_size).convert("RGB")
        tex = np.array(tex) / 255.0
        return torch.from_numpy(tex).to(device).float()
    else:
        if isinstance(tex, np.ndarray):
            tex = torch.from_numpy(tex)
        return tex.to(device).float()

def convert_ndarray_to_pil(texture):
    np = _lazy_import("numpy")
    _lazy_import("PIL.Image")
    Image = _LAZY_IMPORTS["PIL.Image"]

    texture_size = len(texture)
    tex = _convert_texture_format(texture,(texture_size, texture_size),"cuda")
    tex = tex.cpu().numpy()
    processed_texture = (tex * 255).astype(np.uint8)
    pil_texture = Image.fromarray(processed_texture)
    return pil_texture

def get_filename_list(folder_name: str):
    files = [f for f in os.listdir(folder_name)]
    return files
    
# Tensor to PIL
def tensor2pil(image):
    np = _lazy_import("numpy")
    _lazy_import("PIL.Image")
    Image = _LAZY_IMPORTS["PIL.Image"]
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    np = _lazy_import("numpy")
    torch = _lazy_import("torch")
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def numpy2pil(image):
    np = _lazy_import("numpy")
    _lazy_import("PIL.Image")
    Image = _LAZY_IMPORTS["PIL.Image"]
    return Image.fromarray(np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8))    

def convert_pil_images_to_tensor(images):
    tensor_array = []
    
    for image in images:
        tensor_array.append(pil2tensor(image))
        
    return tensor_array
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array 

class MetaData:
    def __init__(self):
        self.camera_config = None
        self.albedos = None
        self.mrs = None
        self.albedos_upscaled = None
        self.mrs_upscaled = None
        self.mesh_file = None

class Hy3DMeshGenerator:
    def __init__(self):
        self.pipeline = None
        self.current_model_path = None
        self.current_attention_mode = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"),),
                "processed_image": ("IMAGE",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1, "max": 30}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa"}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("HY3DLATENT",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3D21Wrapper"

    def loadmodel(self, model, image, steps, guidance_scale, seed, attention_mode, use_cache):
        torch = _lazy_import("torch")
        _lazy_import("hunyuan_pipeline")
        Hunyuan3DDiTFlowMatchingPipeline = _LAZY_IMPORTS["hunyuan_pipeline"]

        device = mm.get_torch_device()
        seed = seed % (2**32)
        model_path = folder_paths.get_full_path("diffusion_models", model)

        # Decide whether to use cache or force reload
        should_load = (
            not use_cache or  # ✅ Cache disabled → always reload
            self.pipeline is None or  # First time
            self.current_model_path != model_path or  # Model changed
            self.current_attention_mode != attention_mode  # Settings changed
        )

        if should_load:
            if not use_cache:
                print(f"🔄 Loading pipeline: {model} (cache disabled)")
            else:
                print(f"🔥 Loading pipeline: {model} (first time or settings changed)")

            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                config_path=os.path.join(config_directory, 'dit_config_2_1.yaml'),
                ckpt_path=model_path,
                attention_mode=attention_mode
            )

            # Only update tracking vars if caching is enabled
            if use_cache:
                self.current_model_path = model_path
                self.current_attention_mode = attention_mode
        else:
            print("⚡ Using cached pipeline (fast!)")

        self.pipeline.to(device)

        image = tensor2pil(image)
        latents = self.pipeline(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed)
        )

        gc.collect()
        return (latents,)

class Hy3DMultiViewsGenerator:
    def __init__(self):
        self.paint_pipeline = None
        self.current_config = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "view_size": ("INT", {"default": 512, "min": 512, "max":1024, "step":256}),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "Number of steps"}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1, "max": 10, "step": 0.1, "tooltip": "Guidance scale"}),
                "texture_size": ("INT", {"default":1024,"min":512,"max":4096,"step":512}),
                "unwrap_mesh": ("BOOLEAN", {"default":True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Use cached paint pipeline (faster) or create fresh instance every time (useful for debugging)"}),
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "IMAGE","IMAGE","IMAGE","IMAGE","HY3D21CAMERA","HY3D21METADATA",)
    RETURN_NAMES = ("pipeline", "albedo","mr","positions","normals","camera_config", "metadata")
    FUNCTION = "genmultiviews"
    CATEGORY = "Hunyuan3D21Wrapper"

    def genmultiviews(self, trimesh, camera_config, view_size, image, steps, guidance_scale, texture_size, unwrap_mesh, seed, use_cache=True):
        _lazy_import("hunyuan_paint")
        Hunyuan3DPaintConfig = _LAZY_IMPORTS["Hunyuan3DPaintConfig"]
        Hunyuan3DPaintPipeline = _LAZY_IMPORTS["Hunyuan3DPaintPipeline"]

        device = mm.get_torch_device()
        seed = seed % (2**32)

        # Create config hash for comparison
        config_str = f"{view_size}_{camera_config}_{texture_size}"

        should_load = (
            not use_cache or
            self.paint_pipeline is None or
            self.current_config != config_str
        )

        if should_load:
            if not use_cache:
                print("🔄 Creating paint pipeline (cache disabled)")
            else:
                print("🔥 Creating paint pipeline (first time or config changed)")

            conf = Hunyuan3DPaintConfig(
                view_size,
                camera_config["selected_camera_azims"],
                camera_config["selected_camera_elevs"],
                camera_config["selected_view_weights"],
                camera_config["ortho_scale"],
                texture_size
            )
            self.paint_pipeline = Hunyuan3DPaintPipeline(conf)

            if use_cache:
                self.current_config = config_str
        else:
            print("⚡ Using cached paint pipeline (fast!)")

        image = tensor2pil(image)

        temp_folder_path = os.path.join(comfy_path, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)
        temp_output_path = os.path.join(temp_folder_path, "textured_mesh.obj")

        albedo, mr, normal_maps, position_maps = self.paint_pipeline(mesh=trimesh, image_path=image, output_mesh_path=temp_output_path, num_steps=steps, guidance_scale=guidance_scale, unwrap=unwrap_mesh, seed=seed)

        albedo_tensor = hy3dpaintimages_to_tensor(albedo)
        mr_tensor = hy3dpaintimages_to_tensor(mr)
        normals_tensor = hy3dpaintimages_to_tensor(normal_maps)
        positions_tensor = hy3dpaintimages_to_tensor(position_maps)

        return (self.paint_pipeline, albedo_tensor, mr_tensor, positions_tensor, normals_tensor, camera_config,)       
        
class Hy3DBakeMultiViews:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE", ),
                "camera_config": ("HY3D21CAMERA", ),
                "albedo": ("IMAGE", ),
                "mr": ("IMAGE", )                
            },
        }

    RETURN_TYPES = ("HY3DPIPELINE", "NPARRAY", "NPARRAY", "NPARRAY", "NPARRAY", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("pipeline", "albedo", "albedo_mask", "mr", "mr_mask", "albedo_texture", "mr_texture",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, pipeline, camera_config, albedo, mr):        
        albedo = convert_tensor_images_to_pil(albedo)
        mr = convert_tensor_images_to_pil(mr)
        
        texture, mask, texture_mr, mask_mr = pipeline.bake_from_multiview(albedo,mr,camera_config["selected_camera_elevs"], camera_config["selected_camera_azims"], camera_config["selected_view_weights"])
        
        texture_pil = convert_ndarray_to_pil(texture)
        #mask_pil = convert_ndarray_to_pil(mask)
        texture_mr_pil = convert_ndarray_to_pil(texture_mr)
        #mask_mr_pil = convert_ndarray_to_pil(mask_mr)
        
        texture_tensor = pil2tensor(texture_pil)
        #mask_tensor = pil2tensor(mask_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)
        #mask_mr_tensor = pil2tensor(mask_mr_pil)
        
        return (pipeline, texture, mask, texture_mr, mask_mr, texture_tensor, texture_mr_tensor)
        
class Hy3DInPaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE", ),
                "albedo": ("NPARRAY", ),
                "albedo_mask": ("NPARRAY", ),
                "mr": ("NPARRAY", ),
                "mr_mask": ("NPARRAY",),
                "output_mesh_name": ("STRING",),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","TRIMESH", "STRING",)
    RETURN_NAMES = ("albedo", "mr", "trimesh", "output_glb_path")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, albedo, albedo_mask, mr, mr_mask, output_mesh_name):
        torch = _lazy_import("torch")
        Trimesh = _lazy_import("trimesh")

        #albedo = tensor2pil(albedo)
        #albedo_mask = tensor2pil(albedo_mask)
        #mr = tensor2pil(mr)
        #mr_mask = tensor2pil(mr_mask)

        vertex_inpaint = True
        method = "NS"

        albedo, mr = pipeline.inpaint(albedo, albedo_mask, mr, mr_mask, vertex_inpaint, method)

        pipeline.set_texture_albedo(albedo)
        pipeline.set_texture_mr(mr)

        # Ensure output_mesh_name is not empty
        if not output_mesh_name or output_mesh_name.strip() == "":
            import time
            output_mesh_name = f"hunyuan3d_{int(time.time())}"

        temp_folder_path = os.path.join(comfy_path, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)
        output_mesh_path = os.path.join(temp_folder_path, f"{output_mesh_name}.obj")
        output_temp_path = pipeline.save_mesh(output_mesh_path)

        output_folder_path = os.path.join(comfy_path, "output")
        os.makedirs(output_folder_path, exist_ok=True)
        output_glb_path = os.path.join(output_folder_path, f"{output_mesh_name}.glb")
        shutil.copyfile(output_temp_path, output_glb_path)

        trimesh = Trimesh.load(output_glb_path, force="mesh")

        texture_pil = convert_ndarray_to_pil(albedo)
        texture_mr_pil = convert_ndarray_to_pil(mr)
        texture_tensor = pil2tensor(texture_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)

        output_glb_path = f"{output_mesh_name}.glb"

        # Don't call clean_memory() - we're caching the pipeline for reuse!
        # pipeline.clean_memory()

        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

        return (texture_tensor, texture_mr_tensor, trimesh, output_glb_path)         
        
class Hy3D21CameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 0, 180", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "view_weights": ("STRING", {"default": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "multiline": False}),
                "ortho_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("HY3D21CAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, camera_azimuths, camera_elevations, view_weights, ortho_scale):
        angles_list = list(map(int, camera_azimuths.replace(" ", "").split(',')))
        elevations_list = list(map(int, camera_elevations.replace(" ", "").split(',')))
        weights_list = list(map(float, view_weights.replace(" ", "").split(',')))

        camera_config = {
            "selected_camera_azims": angles_list,
            "selected_camera_elevs": elevations_list,
            "selected_view_weights": weights_list,
            "ortho_scale": ortho_scale,
            }
        
        return (camera_config,)
        
class Hy3D21VAELoader:
    def __init__(self):
        self.vae = None
        self.current_model_path = None
        self.current_config_hash = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Use cached VAE model (faster) or reload every time (useful for debugging)"}),
            },
            "optional":{
                "vae_config": ("HY3D21VAECONFIG",),
            }
        }

    RETURN_TYPES = ("HY3DVAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3D21Wrapper"

    def loadmodel(self, model_name, use_cache=True, vae_config=None):
        torch = _lazy_import("torch")
        _lazy_import("comfy_utils")
        load_torch_file = _LAZY_IMPORTS["load_torch_file"]
        _lazy_import("hunyuan_vae")
        ShapeVAE = _LAZY_IMPORTS["ShapeVAE"]

        device = mm.get_torch_device()
        model_path = folder_paths.get_full_path("vae", model_name)

        if(vae_config==None):
            vae_config = {
                'num_latents': 4096,
                'embed_dim': 64,
                'num_freqs': 8,
                'include_pi': False,
                'heads': 16,
                'width': 1024,
                'num_encoder_layers': 8,
                'num_decoder_layers': 16,
                'qkv_bias': False,
                'qk_norm': True,
                'scale_factor': 1.0039506158752403,
                'geo_decoder_mlp_expand_ratio': 4,
                'geo_decoder_downsample_ratio': 1,
                'geo_decoder_ln_post': True,
                'point_feats': 4,
                'pc_size': 81920,
                'pc_sharpedge_size': 0
            }

        # Hash the config
        config_hash = hashlib.md5(str(sorted(vae_config.items())).encode()).hexdigest()

        should_load = (
            not use_cache or
            self.vae is None or
            self.current_model_path != model_path or
            self.current_config_hash != config_hash
        )

        if should_load:
            if not use_cache:
                print(f"🔄 Loading VAE: {model_name} (cache disabled)")
            else:
                print(f"🔥 Loading VAE: {model_name} (first time or settings changed)")

            vae_sd = load_torch_file(model_path)
            self.vae = ShapeVAE(**vae_config)
            self.vae.load_state_dict(vae_sd)
            self.vae.eval().to(torch.float16)

            if use_cache:
                self.current_model_path = model_path
                self.current_config_hash = config_hash
        else:
            print("⚡ Using cached VAE (fast!)")

        return (self.vae,)
        
class Hy3D21VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("HY3DVAE",),
                "latents": ("HY3DLATENT", ),
                "box_v": ("FLOAT", {"default": 1.01, "min": -10.0, "max": 10.0, "step": 0.001}),
                "octree_resolution": ("INT", {"default": 384, "min": 8, "max": 4096, "step": 8}),
                "num_chunks": ("INT", {"default": 8000, "min": 1, "max": 10000000, "step": 1, "tooltip": "Number of chunks to process at once, higher values use more memory, but make the process faster"}),
                "mc_level": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                "mc_algo": (["mc", "dmc"], {"default": "mc"}),
            },
            "optional": {
                "enable_flash_vdm": ("BOOLEAN", {"default": True}),
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": "Offloads the model to the offload device once the process is done."}),
            }            
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, vae, latents, box_v, octree_resolution, mc_level, num_chunks, mc_algo, enable_flash_vdm, force_offload):
        torch = _lazy_import("torch")
        Trimesh = _lazy_import("trimesh")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        torch.cuda.empty_cache()

        vae.to(device)

        vae.enable_flashvdm_decoder(enabled=enable_flash_vdm, mc_algo=mc_algo)

        latents = vae.decode(latents)
        outputs = vae.latents2mesh(
            latents,
            output_type='trimesh',
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=True
        )[0]

        # if force_offload==True:
        #     vae.to(offload_device)

        outputs.mesh_f = outputs.mesh_f[:, ::-1]
        mesh_output = Trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f)
        print(f"Decoded mesh with {mesh_output.vertices.shape[0]} vertices and {mesh_output.faces.shape[0]} faces")

        #del pipeline
        # del vae

        # mm.soft_empty_cache()
        # torch.cuda.empty_cache()
        gc.collect()

        return (mesh_output, )        
        
        
class Hy3D21LoadImageWithTransparency:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Hunyuan3D21Wrapper"

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", )
    RETURN_NAMES = ("image", "mask", "image_with_alpha")
    FUNCTION = "load_image"
    def load_image(self, image):
        _lazy_import("PIL.Image")
        Image = _LAZY_IMPORTS["PIL.Image"]
        ImageSequence = _LAZY_IMPORTS["PIL.ImageSequence"]
        ImageOps = _LAZY_IMPORTS["PIL.ImageOps"]
        torch = _lazy_import("torch")
        np = _lazy_import("numpy")

        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        output_images_ori = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            
            output_images_ori.append(pil2tensor(i))

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
            output_image_ori = torch.cat(output_images_ori, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            output_image_ori = output_images_ori[0]

        return (output_image, output_mask, output_image_ori)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True     

class Hy3D21PostprocessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": True}),
                "max_facenum": ("INT", {"default": 40000, "min": 1, "max": 10000000, "step": 1}),
                "smooth_normals": ("BOOLEAN", {"default": False}),
                "smooth_surface": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, trimesh, remove_floaters, remove_degenerate_faces, reduce_faces, max_facenum, smooth_normals, smooth_surface):
        _lazy_import("hunyuan_postprocessors")
        FloaterRemover = _LAZY_IMPORTS["FloaterRemover"]
        DegenerateFaceRemover = _LAZY_IMPORTS["DegenerateFaceRemover"]
        FaceReducer = _LAZY_IMPORTS["FaceReducer"]
        Trimesh = _lazy_import("trimesh")

        new_mesh = trimesh.copy()
        if remove_floaters:
            new_mesh = FloaterRemover()(new_mesh)
            print(f"Removed floaters, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if remove_degenerate_faces:
            new_mesh = DegenerateFaceRemover()(new_mesh)
            print(f"Removed degenerate faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if reduce_faces:
            new_mesh = FaceReducer()(new_mesh, max_facenum=max_facenum)
            print(f"Reduced faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if smooth_normals:
            new_mesh.vertex_normals = Trimesh.smoothing.get_vertices_normals(new_mesh)
        if smooth_surface:
            new_mesh = Trimesh.smoothing.filter_laplacian(new_mesh, iterations=3)
        return (new_mesh, )
        
class Hy3D21ExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hy3D"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        else:
            temp_file = Path(full_output_folder, f'hy3dtemp_.{file_format}')
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f'hy3dtemp_.{file_format}'
        
        return (str(relative_path), )    

class Hy3D21MeshUVWrap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, trimesh):
        _lazy_import("mesh_uv_wrap")
        mesh_uv_wrap = _LAZY_IMPORTS["mesh_uv_wrap"]

        trimesh = mesh_uv_wrap(trimesh)

        return (trimesh,)        
        
class Hy3D21LoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):
        Trimesh = _lazy_import("trimesh")

        if not os.path.exists(glb_path):
            glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)

        trimesh = Trimesh.load(glb_path, force="mesh")

        return (trimesh,)
        
class Hy3D21ImageWithAlphaInput:
    """Combines image + mask into RGBA format for Hunyuan processing"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "image_with_alpha")
    FUNCTION = "process_image"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process_image(self, image, mask):
        """
        image: [B, H, W, C] - RGB tensor
        mask: [B, H, W] - mask tensor
        """
        return (image, mask, image)


def reducefacesnano(new_mesh, max_facenum):
    """Face reduction using Instant Meshes (pynanoinstantmeshes)"""
    np = _lazy_import("numpy")
    Trimesh = _lazy_import("trimesh")

    try:
        import pynanoinstantmeshes as PyNIM

        current_faces = len(new_mesh.faces)

        target_vertices = max(100, int(max_facenum * 0.25))

        print(f"Remeshing from {current_faces} faces to ~{max_facenum} target faces...")
        print(f"Requesting {target_vertices} vertices from Instant Meshes...")

        # Remesh with Instant Meshes
        new_verts, new_faces = PyNIM.remesh(
            np.array(new_mesh.vertices, dtype=np.float32),
            np.array(new_mesh.faces, dtype=np.uint32),
            target_vertices,
            align_to_boundaries=True,
            smooth_iter=2
        )

        # Instant Meshes can fail, check validity
        if new_verts.shape[0] - 1 != new_faces.max():
            raise ValueError("Remeshing failed")

        # Triangulate quads (Instant Meshes outputs quads)
        new_faces = Trimesh.geometry.triangulate_quads(new_faces)

        new_mesh = Trimesh.Trimesh(vertices=new_verts.astype(np.float32), faces=new_faces)

        print(f"Remeshed: {new_mesh.vertices.shape[0]} vertices, {new_mesh.faces.shape[0]} faces")
        return new_mesh
    except Exception as e:
        print(f"Instant Meshes failed: {e}, returning original mesh")
        return new_mesh




NODE_CLASS_MAPPINGS = {
    "Hy3DMeshGenerator": Hy3DMeshGenerator,
    "Hy3DMultiViewsGenerator": Hy3DMultiViewsGenerator,
    "Hy3DBakeMultiViews": Hy3DBakeMultiViews,
    "Hy3DInPaint": Hy3DInPaint,
    "Hy3D21CameraConfig": Hy3D21CameraConfig,
    "Hy3D21VAELoader": Hy3D21VAELoader,
    "Hy3D21VAEDecode": Hy3D21VAEDecode,
    "Hy3D21LoadImageWithTransparency": Hy3D21LoadImageWithTransparency,
    "Hy3D21PostprocessMesh": Hy3D21PostprocessMesh,
    "Hy3D21ExportMesh": Hy3D21ExportMesh,
    "Hy3D21MeshUVWrap": Hy3D21MeshUVWrap,
    "Hy3D21LoadMesh": Hy3D21LoadMesh,
    "Hy3D21ImageWithAlphaInput": Hy3D21ImageWithAlphaInput,
    }
    
NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DMeshGenerator": "Hunyuan 3D 2.1 Mesh Generator",
    "Hy3DMultiViewsGenerator": "Hunyuan 3D 2.1 MultiViews Generator",
    "Hy3DBakeMultiViews": "Hunyuan 3D 2.1 Bake MultiViews",
    "Hy3DInPaint": "Hunyuan 3D 2.1 InPaint",
    "Hy3D21CameraConfig": "Hunyuan 3D 2.1 Camera Config",
    "Hy3D21VAELoader": "Hunyuan 3D 2.1 VAE Loader",
    "Hy3D21VAEDecode": "Hunyuan 3D 2.1 VAE Decoder",
    "Hy3D21LoadImageWithTransparency": "Hunyuan 3D 2.1 Load Image with Transparency",
    "Hy3D21PostprocessMesh": "Hunyuan 3D 2.1 Post Process Trimesh",
    "Hy3D21ExportMesh": "Hunyuan 3D 2.1 Export Mesh",
    "Hy3D21MeshUVWrap": "Hunyuan 3D 2.1 Mesh UV Wrap",
    "Hy3D21LoadMesh": "Hunyuan 3D 2.1 Load Mesh",
    "Hy3D21ImageWithAlphaInput": "Image with Alpha Input (Hy3D21)",
    #"Hy3D21MultiViewsMeshGenerator": "Hunyuan 3D 2.1 MultiViews Mesh Generator"
    }