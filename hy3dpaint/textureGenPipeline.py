 # Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import copy
import trimesh
import numpy as np
import gc
from PIL import Image
from typing import List
from .DifferentiableRenderer.MeshRender import MeshRender
from .utils.simplify_mesh_utils import remesh_mesh
from .utils.multiview_utils import multiviewDiffusionNet
from .utils.pipeline_utils import ViewProcessor
#from .utils.image_super_utils import imageSuperNet
from .utils.uvwrap_utils import mesh_uv_wrap
from .convert_utils import create_glb_with_pbr_materials
#from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
import warnings
import folder_paths
import comfy.model_management as mm

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)

def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> bool:
    # 执行转换
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)

class Hunyuan3DPaintConfig:
    def __init__(self, resolution, camera_azims, camera_elevs, view_weights, ortho_scale, texture_size):
        self.device = "cuda"

        cfg_path = os.path.join(
            os.path.dirname(__file__), "cfgs", "hunyuan-paint-pbr.yaml"
        )

        self.multiview_cfg_path = cfg_path
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024
        self.texture_size = texture_size
        self.max_selected_view_num = 32
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"
        self.ortho_scale = ortho_scale

        # view selection
        # self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        # self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        # self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        self.candidate_camera_azims = camera_azims
        self.candidate_camera_elevs = camera_elevs
        self.candidate_view_weights = view_weights

        # for azim in range(0, 360, 30):
            # self.candidate_camera_azims.append(azim)
            # self.candidate_camera_elevs.append(20)
            # self.candidate_view_weights.append(0.01)

            # self.candidate_camera_azims.append(azim)
            # self.candidate_camera_elevs.append(-20)
            # self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:
    _cached_multiview_model = None
    _cached_model_config = None

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.model = None
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
            ortho_scale=self.config.ortho_scale
        )
        self.view_processor = ViewProcessor(self.config, self.render)

    def load_models(self):
        torch.cuda.empty_cache()
        if not hasattr(self, 'model') or self.model is None:
                self.model = self._get_or_load_model()
        print("Models Loaded.")

    def _get_or_load_model(self):
        """Get cached model or load if not cached"""
        device = mm.get_torch_device()  # Add this
        
        # Create a config key for caching (use relevant config parameters)
        config_key = (
            self.config.resolution,
            self.config.texture_size,
            # Add other config params that affect model loading
        )
        
        # Check if we need to load/reload the model
        if (Hunyuan3DPaintPipeline._cached_multiview_model is None or
            Hunyuan3DPaintPipeline._cached_model_config != config_key):
            
            print("🔥 Loading multiview diffusion model (first time or config changed)")
            Hunyuan3DPaintPipeline._cached_multiview_model = multiviewDiffusionNet(self.config)
            Hunyuan3DPaintPipeline._cached_model_config = config_key
        else:
            print("⚡ Using cached multiview model")
        
        # CRITICAL: Ensure model is on the correct device
        model = Hunyuan3DPaintPipeline._cached_multiview_model
        # Move the pipeline components to the correct device
        if hasattr(model, 'pipeline'):
            model.pipeline.to(device)
        
        return model

    @torch.no_grad()
    def __call__(self, mesh, image_path=None, output_mesh_path=None, use_remesh=False, save_glb=True, num_steps=10, guidance_scale=3.0, unwrap=True, seed=0):
        """Generate texture for 3D mesh using multiview diffusion"""
        if not hasattr(self, 'model') or self.model is None:
            self.model = self._get_or_load_model()        
        # Ensure image_prompt is a list
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        if not isinstance(image_prompt, List):
            image_prompt = [image_prompt]
        else:
            image_prompt = image_path

        # Process mesh
        # path = os.path.dirname(mesh_path)
        # if use_remesh:
            # processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            # remesh_mesh(mesh_path, processed_mesh_path)
            # print('Mesh Simplified')
        # else:
            # processed_mesh_path = mesh_path

        # Output path
        # if output_mesh_path is None:
            # output_mesh_path = os.path.join(path, f"textured_mesh.obj")

        # Load mesh
        #mesh = trimesh.load(processed_mesh_path)
        if(unwrap==True):
            print('Unwrapping Mesh ...')
            mesh = mesh_uv_wrap(mesh)
            
        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        # selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            # self.config.candidate_camera_elevs,
            # self.config.candidate_camera_azims,
            # self.config.candidate_view_weights,
            # self.config.max_selected_view_num,
        # )
        
        selected_camera_elevs = self.config.candidate_camera_elevs
        selected_camera_azims = self.config.candidate_camera_azims
        selected_view_weights = self.config.candidate_view_weights

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)

        ##########  Style  ###########
        image_caption = "high quality"
        # image_style = []
        # for image in image_prompt:
            # if image.mode == "RGBA":
                # white_bg = Image.new("RGB", image.size, (255, 255, 255))
                # white_bg.paste(image, mask=image.getchannel("A"))
                # image = white_bg
            # image_style.append(image)
        # image_style = [image.convert("RGB") for image in image_style]

        mm.soft_empty_cache()
        torch.cuda.empty_cache()

        ###########  Multiview  ##########
        print('Generating MultiViews PBR ...')
        multiviews_pbr = self.model(
            image_prompt,
            normal_maps + position_maps,
            prompt=image_caption,
            custom_view_size=self.config.resolution,
            resize_input=True,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        return multiviews_pbr["albedo"], multiviews_pbr["mr"], normal_maps, position_maps
        
        ###########  Enhance  ##########
        # enhance_images = {}
        # enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
        # enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])

        # print('Upscaling Views ...')
        # for i in range(len(enhance_images["albedo"])):
            # enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            # enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])

        ###########  Bake  ##########
        # print('Baking Views ...')
        # for i in range(len(multiviews_pbr["albedo"])):
            # multiviews_pbr["albedo"][i] = multiviews_pbr["albedo"][i].resize(
                # (self.config.render_size, self.config.render_size)
            # )
            # multiviews_pbr["mr"][i] = multiviews_pbr["mr"][i].resize((self.config.render_size, self.config.render_size))
        # texture, mask = self.view_processor.bake_from_multiview(
            # multiviews_pbr["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        # )
        # mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        # texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            # multiviews_pbr["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        # )
        # mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        # #########  inpaint  ###########
        # print('Inpainting ...')
        # texture = self.view_processor.texture_inpaint(texture, mask_np)
        # self.render.set_texture(texture, force_set=True)
        # if "mr" in multiviews_pbr:
            # texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            # self.render.set_texture_mr(texture_mr)

        # self.render.save_mesh(output_mesh_path, downsample=True)

        # if save_glb:
            # output_glb_path = output_mesh_path.replace(".obj", ".glb")
            # conversion_success = quick_convert_with_obj2gltf(output_mesh_path, output_glb_path)            
            
            # mesh = Trimesh.load(output_mesh_path, force="mesh")
            # output_glb_path = output_mesh_path.replace(".obj", ".glb")
            # mesh.export(output_glb_path, file_type='glb')
            # convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
            # output_glb_path = output_mesh_path.replace(".obj", ".glb")

        # return output_mesh_path
        
    def set_texture_albedo(self, texture):
        self.render.set_texture(texture, force_set=True)
        
    def set_texture_mr(self, texture_mr):
        self.render.set_texture_mr(texture_mr)
        
    def save_mesh(self, output_mesh_path):
        self.render.save_mesh(output_mesh_path, downsample=False)
        output_glb_path = output_mesh_path.replace(".obj", ".glb")
        conversion_success = quick_convert_with_obj2gltf(output_mesh_path, output_glb_path)
        
        return output_glb_path
        
    def inpaint(self, albedo, albedo_mask, mr, mr_mask, vertex_inpaint, method):
        #mask_np = np.asarray(albedo)
        mask_np = (albedo_mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = self.view_processor.texture_inpaint(albedo, mask_np, vertex_inpaint, method)
        
        mask_mr_np = (mr_mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        #mask_mr_np = np.asarray(mr_mask)
        texture_mr = self.view_processor.texture_inpaint(mr, mask_mr_np, vertex_inpaint, method)
        
        return texture, texture_mr
        
    def bake_from_multiview(self, albedo, mr, selected_camera_elevs, selected_camera_azims, selected_view_weights):
        texture, mask = self.view_processor.bake_from_multiview(
            albedo, selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            mr, selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        
        return texture, mask, texture_mr, mask_mr
        
    def clean_memory(self):
        del self.render
        del self.view_processor
        del self.model
        
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()    
        
    def load_mesh(self, mesh):
        self.render.load_mesh(mesh=mesh)
        
        
